#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_hermes_json_call() {
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "{\"name\": \"get_weather\", \"arguments\": "
                                                   "{\"city\": \"Paris\", \"days\": 2}}\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "hermes call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "hermes content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed hermes call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "hermes call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "hermes function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "hermes string argument parsed");
    failures += check(args.at("days") == 2, "hermes number argument parsed");
    return failures;
}

// The two dialects are chosen per call by the model, not per response, so a
// single turn can legitimately mix them.
int test_hermes_and_xml_mix() {
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n{\"name\":\"first\",\"arguments\":{\"value\":1}}\n</tool_call>\n"
        "<tool_call>\n<function=second>\n<parameter=value>\n2\n</parameter>\n</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "mixed dialects parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed mixed calls");
    if (parsed.tool_calls.size() != 2) { return failures; }
    failures += check(parsed.tool_calls[0].name == "first", "mixed first call name");
    failures += check(parsed.tool_calls[1].name == "second", "mixed second call name");
    failures += check(Json::parse(parsed.tool_calls[0].arguments_json).at("value") == 1,
                      "mixed hermes argument");
    failures += check(Json::parse(parsed.tool_calls[1].arguments_json).at("value") == 2,
                      "mixed xml parameter");
    return failures;
}

int test_hermes_argument_shapes() {
    const ninfer::serve::ParsedToolCallOutput encoded = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n{\"name\":\"call\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n"
        "</tool_call>",
        64);
    const ninfer::serve::ParsedToolCallOutput absent = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n{\"name\":\"ping\"}\n</tool_call>", 64);
    const ninfer::serve::ParsedToolCallOutput null_args =
        ninfer::serve::parse_qwen_tool_call_output(
            "<tool_call>\n{\"name\":\"ping\",\"arguments\":null}\n</tool_call>", 64);

    int failures = 0;
    failures += check(encoded.is_tool_call_response && encoded.tool_calls.size() == 1 &&
                          Json::parse(encoded.tool_calls[0].arguments_json).at("city") == "Paris",
                      "double-encoded arguments string decoded");
    failures += check(absent.is_tool_call_response && absent.tool_calls.size() == 1 &&
                          absent.tool_calls[0].arguments_json == "{}",
                      "absent arguments become an empty object");
    failures += check(null_args.is_tool_call_response && null_args.tool_calls.size() == 1 &&
                          null_args.tool_calls[0].arguments_json == "{}",
                      "null arguments become an empty object");
    return failures;
}

// Fallback has to stay total: the streaming filter replays the buffered region
// verbatim, so anything the parser rejects must survive as the original bytes.
int test_hermes_rejections_fall_back_to_text() {
    int failures = 0;
    for (const char* text : {
             "<tool_call>\n{\"name\":\"get_weather\",\"argum",
             "<tool_call>\n[\"get_weather\"]\n</tool_call>",
             "<tool_call>\n{\"arguments\":{}}\n</tool_call>",
             "<tool_call>\n{\"name\":\"bad name!\",\"arguments\":{}}\n</tool_call>",
             "<tool_call>\n{\"name\":\"ok\",\"arguments\":[1,2]}\n</tool_call>",
             "<tool_call>\n{\"name\":\"ok\",\"arguments\":{}} tail\n</tool_call>",
         }) {
        const ninfer::serve::ParsedToolCallOutput parsed =
            ninfer::serve::parse_qwen_tool_call_output(text, 64);
        failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                              parsed.content == text,
                          std::string("rejected hermes body preserved verbatim: ") + text);
    }
    return failures;
}

int test_hermes_honours_name_limit() {
    const std::string text =
        "<tool_call>\n{\"name\":\"" + std::string(128, 'a') + "\",\"arguments\":{}}\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response, "hermes 128-character name accepted at 128");
    failures += check(!openai.is_tool_call_response, "hermes 128-character name rejected at 64");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const ninfer::serve::ParsedToolCallOutput parsed =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const ninfer::serve::ParsedToolCallOutput anthropic =
        ninfer::serve::parse_qwen_tool_call_output(text, 128);
    const ninfer::serve::ParsedToolCallOutput openai =
        ninfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const ninfer::serve::ParsedToolCallOutput too_long =
        ninfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    ninfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    ninfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    ninfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

} // namespace

ninfer::serve::ToolDefinition typed_tool(const std::string& name, const Json& properties) {
    ninfer::serve::ToolDefinition tool;
    tool.name            = name;
    tool.parameters_json = Json{{"type", "object"}, {"properties", properties}}.dump();
    return tool;
}

int test_declared_types_shape_values() {
    const std::vector<ninfer::serve::ToolDefinition> tools = {typed_tool(
        "ship", Json{{"items", Json{{"type", "array"}}},
                     {"express", Json{{"type", "boolean"}}},
                     {"note", Json{{"type", "string"}}},
                     {"code", Json{{"type", "string"}}},
                     {"limit", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"type", "null"}}})}}},
                     {"flag", Json{{"type", "boolean"}}}})};
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n<function=ship>\n"
        "<parameter=items>\n[{\"sku\":\"SKU-11\",\"qty\":2}]\n</parameter>\n"
        "<parameter=express>\nTrue\n</parameter>\n"
        "<parameter=note>\n{\"looks\":\"like json\"}\n</parameter>\n"
        "<parameter=code>\n  indented\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n"
        "<parameter=flag>\n\n</parameter>\n"
        "</function>\n</tool_call>",
        64, tools);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "typed call parsed as tool response");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("items").at(0).at("qty") == 2, "array parameter parsed as JSON");
    failures += check(args.at("express") == true, "capitalized boolean accepted");
    failures += check(args.at("note") == "{\"looks\":\"like json\"}",
                      "string parameter kept as text although it parses as JSON");
    failures += check(args.at("code") == "  indented", "string keeps its own indentation");
    failures += check(args.at("limit") == 5, "anyOf integer parsed as number");
    failures += check(!args.contains("flag"), "empty boolean left out");
    return failures;
}

int test_undeclared_parameters_keep_guessing() {
    const std::vector<ninfer::serve::ToolDefinition> tools = {
        typed_tool("ship", Json{{"note", Json{{"type", "string"}}}})};
    const ninfer::serve::ParsedToolCallOutput parsed = ninfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n<function=ship>\n<parameter=count>\n3\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=other>\n<parameter=note>\n7\n</parameter>\n"
        "</function>\n</tool_call>",
        64, tools);

    int failures = 0;
    failures += check(parsed.tool_calls.size() == 2, "two calls with partial schemas");
    if (parsed.tool_calls.size() != 2) { return failures; }
    failures += check(Json::parse(parsed.tool_calls[0].arguments_json).at("count") == 3,
                      "parameter missing from the schema is guessed");
    failures += check(Json::parse(parsed.tool_calls[1].arguments_json).at("note") == 7,
                      "tool missing from the list is guessed");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_declared_types_shape_values();
    failures += test_undeclared_parameters_keep_guessing();
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_hermes_json_call();
    failures += test_hermes_and_xml_mix();
    failures += test_hermes_argument_shapes();
    failures += test_hermes_rejections_fall_back_to_text();
    failures += test_hermes_honours_name_limit();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

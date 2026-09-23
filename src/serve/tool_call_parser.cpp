#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

// JSON Schema types a parameter declares at its top level, through "type" or the alternatives of
// "anyOf"/"oneOf". Empty means nothing usable is declared and the value is guessed from its text.
struct DeclaredTypes {
    bool any     = false;
    bool string  = false;
    bool boolean = false;
    bool integer = false;
    bool number  = false;
    bool object  = false;
    bool array   = false;
    bool null    = false;
};

void collect_types(const Json& schema, DeclaredTypes& types) {
    if (!schema.is_object()) { return; }
    const auto add = [&types](const Json& name) {
        if (!name.is_string()) { return; }
        const std::string& type = name.get_ref<const std::string&>();
        bool* slot              = type == "string"    ? &types.string
                                  : type == "boolean" ? &types.boolean
                                  : type == "integer" ? &types.integer
                                  : type == "number"  ? &types.number
                                  : type == "object"  ? &types.object
                                  : type == "array"   ? &types.array
                                  : type == "null"    ? &types.null
                                                      : nullptr;
        if (slot != nullptr) {
            *slot     = true;
            types.any = true;
        }
    };
    if (const auto type = schema.find("type"); type != schema.end()) {
        if (type->is_array()) {
            for (const Json& name : *type) { add(name); }
        } else {
            add(*type);
        }
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        const auto alternatives = schema.find(key);
        if (alternatives == schema.end() || !alternatives->is_array()) { continue; }
        for (const Json& alternative : *alternatives) { collect_types(alternative, types); }
    }
}

DeclaredTypes declared_types(const Json* properties, const std::string& key) {
    DeclaredTypes types;
    if (properties == nullptr) { return types; }
    const auto schema = properties->find(key);
    if (schema != properties->end()) { collect_types(*schema, types); }
    return types;
}

bool ascii_case_equal(std::string_view left, std::string_view right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(), [](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) ==
                      std::tolower(static_cast<unsigned char>(b));
           });
}

bool admits(const DeclaredTypes& types, const Json& value) {
    switch (value.type()) {
    case Json::value_t::null:
        return types.null;
    case Json::value_t::boolean:
        return types.boolean;
    case Json::value_t::number_integer:
    case Json::value_t::number_unsigned:
        return types.integer || types.number;
    case Json::value_t::number_float:
        return types.number ||
               (types.integer && std::trunc(value.get<double>()) == value.get<double>());
    case Json::value_t::string:
        return types.string;
    case Json::value_t::object:
        return types.object;
    case Json::value_t::array:
        return types.array;
    default:
        return false;
    }
}

// The template renders a value between a newline after the open tag and one before the close tag;
// those two are framing, everything else belongs to the value.
std::string_view remove_framing_newlines(std::string_view value) {
    if (value.starts_with("\r\n")) {
        value.remove_prefix(2);
    } else if (value.starts_with('\n')) {
        value.remove_prefix(1);
    }
    if (value.ends_with("\r\n")) {
        value.remove_suffix(2);
    } else if (value.ends_with('\n')) {
        value.remove_suffix(1);
    }
    return value;
}

// Qwen writes every argument as bare text, so its type comes from the tool's schema: a string
// parameter keeps the text as is even when it looks like JSON, and the others are read as JSON,
// with True/FALSE accepted for booleans. An empty value of a non-string parameter is left out
// rather than sent as "". Without a declared type the value is guessed from its text as before.
// Returns false when the parameter is to be omitted.
bool normalize_value(std::string_view encoded, const DeclaredTypes& types, Json& value) {
    if (!types.any) {
        const std::string raw = trim_ascii(encoded);
        Json parsed           = Json::parse(raw, nullptr, false);
        value                 = parsed.is_discarded() ? Json(raw) : std::move(parsed);
        return true;
    }
    const std::string_view framed = remove_framing_newlines(encoded);
    if (types.string) {
        value = Json(std::string(framed));
        return true;
    }
    const std::string raw = trim_ascii(framed);
    if (raw.empty()) { return false; }
    Json parsed = Json::parse(raw, nullptr, false);
    if (!parsed.is_discarded() && admits(types, parsed)) {
        value = std::move(parsed);
        return true;
    }
    if (types.boolean && (ascii_case_equal(raw, "true") || ascii_case_equal(raw, "false"))) {
        value = ascii_case_equal(raw, "true");
        return true;
    }
    // Not what the schema asks for: hand it over as parsed, or as text, and let the client's
    // validation answer the model.
    value = parsed.is_discarded() ? Json(std::string(framed)) : std::move(parsed);
    return true;
}

bool parse_parameter(std::string_view inner, std::size_t& pos, const Json* properties, Json& args) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    Json value;
    if (normalize_value(inner.substr(pos, value_end - pos), declared_types(properties, key),
                        value)) {
        args[key] = std::move(value);
    }
    pos = value_end + kParamClose.size();
    return true;
}

// Properties of the named tool's parameter schema, or null when the tool or its schema is unknown.
Json tool_properties(std::span<const ToolDefinition> tools, std::string_view name) {
    for (const ToolDefinition& tool : tools) {
        if (tool.name != name) { continue; }
        Json schema = Json::parse(tool.parameters_json, nullptr, false);
        if (!schema.is_object()) { return nullptr; }
        const auto properties = schema.find("properties");
        if (properties == schema.end() || !properties->is_object()) { return nullptr; }
        return std::move(*properties);
    }
    return nullptr;
}

bool parse_xml_tool_call(std::string_view block, std::size_t max_name_length,
                         std::span<const ToolDefinition> tools, ToolCall& out) {
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const std::string_view params = block.substr(pos, function_end - pos);
    const Json properties         = tool_properties(tools, name);
    const Json* declared          = properties.is_object() ? &properties : nullptr;
    Json args                     = Json::object();
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, declared, args)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

// Qwen3 weights are trained on the Hermes convention, so the model emits a JSON
// body inside <tool_call> even when the prompt teaches the XML dialect above.
// Accepting only XML turned those calls back into prose, which reads to a client
// as a turn that ended without calling anything.
bool parse_hermes_tool_call(std::string_view block, std::size_t max_name_length, ToolCall& out) {
    const Json parsed = Json::parse(trim_ascii(block), nullptr, false);
    if (!parsed.is_object()) { return false; }

    const auto name_field = parsed.find("name");
    if (name_field == parsed.end() || !name_field->is_string()) { return false; }
    const std::string name = name_field->get<std::string>();
    if (!valid_function_name(name, max_name_length)) { return false; }

    Json args                  = Json::object();
    const auto arguments_field = parsed.find("arguments");
    if (arguments_field != parsed.end() && !arguments_field->is_null()) {
        if (arguments_field->is_object()) {
            args = *arguments_field;
        } else if (arguments_field->is_string()) {
            // Double-encoded arguments: the object arrives as a JSON string.
            args = Json::parse(arguments_field->get<std::string>(), nullptr, false);
            if (!args.is_object()) { return false; }
        } else {
            return false;
        }
    }

    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

// Both dialects are all-or-nothing and write to `out` only once they have
// committed, so trying one after the other cannot leave a half-filled call. Hermes arguments
// arrive as JSON already, so only the XML dialect needs the schema to type its values.
bool parse_one_tool_call(std::string_view block, std::size_t max_name_length,
                         std::span<const ToolDefinition> tools, ToolCall& out) {
    return parse_xml_tool_call(block, max_name_length, tools, out) ||
           parse_hermes_tool_call(block, max_name_length, out);
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 std::span<const ToolDefinition> tools) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kToolOpen)) { return fallback(text); }
        const std::size_t inner_begin = pos + kToolOpen.size();
        const std::size_t close       = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) { return fallback(text); }
        ToolCall call;
        if (!parse_one_tool_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                 max_tool_name_length, tools, call)) {
            return fallback(text);
        }
        out.tool_calls.push_back(std::move(call));
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    pending_.append(text);
    const std::size_t marker = pending_.find(kToolOpen);
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        std::string visible = pending_.substr(0, safe_end);
        tool_region_        = pending_.substr(safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        emitted_bytes_ += visible.size();
        return visible;
    }

    const std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(bool is_tool_call_response) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (is_tool_call_response) {
        pending_.clear();
        tool_region_.clear();
        return {};
    }
    std::string tail = std::move(pending_);
    tail += tool_region_;
    tool_region_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace ninfer::serve

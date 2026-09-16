#pragma once

#include <ninfer/targets/qwen3_8/prepared_prompt.h>
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8::frontend_internal {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;
    // Client-selected prompt-cache breakpoint at the end of this part.
    bool cache_breakpoint = false;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    std::string role;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    // part_ends, when given, receives the byte offset in the returned string at which each part
    // ends, in part order.
    [[nodiscard]] std::string rendered_content(bool add_vision_id = false,
                                               int* image_count   = nullptr,
                                               int* video_count   = nullptr,
                                               std::vector<std::size_t>* part_ends = nullptr) const;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    PrefixCheckpointPolicy prefix_checkpoint_policy = PrefixCheckpointPolicy::RollingTool;
    bool add_vision_id                              = false;
    PromptCacheMode prompt_cache_mode               = PromptCacheMode::Implicit;
    std::vector<std::string> tool_jsons;
};

struct PromptBoundaryByteHint {
    PromptBoundaryKind kind = PromptBoundaryKind::SystemTools;
    std::size_t byte_offset = 0;
    bool publish            = false;
};

struct RenderedChat {
    std::string text;
    std::optional<std::size_t> turn_rewrite_byte_offset;
    // Opener of the last real user query. Unlike the turn-rewrite frontier this sits *before*
    // that message's content, so it survives a client rewriting the message's tail - the
    // floating-reminder pattern that otherwise invalidates every deeper anchor.
    std::optional<std::size_t> user_turn_byte_offset;
    // Strictly ascending by byte offset, none past `turn_rewrite_byte_offset`. Template-derived
    // kinds sit on exact token frontiers; Explicit ones sit where the client put them and are
    // snapped to a token boundary when encoded.
    std::vector<PromptBoundaryByteHint> boundaries;
};

enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source);

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}

    ChatTemplateSemantics semantics_;
};

} // namespace ninfer::targets::qwen3_8::frontend_internal

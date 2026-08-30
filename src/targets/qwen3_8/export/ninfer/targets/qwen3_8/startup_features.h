#pragma once

#include "ninfer/ops/lora.h"
#include "ninfer/types.h"

namespace ninfer::targets::qwen3_8 {

struct StartupFeatures {
    bool vision                    = false;
    std::uint32_t vision_max_tokens = 8192;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // Startup-fixed number of resident LoRA slots. Zero means the LoRA leaves are never emitted,
    // so the captured graph is topologically identical to an engine built without adapter
    // support. The pool an engine can select from is unbounded and does not appear here: it
    // changes which bytes a slot holds, never the schedule.
    std::uint32_t lora_slots = 0;
    // Rank the workspace is sized for. The layout is frozen before any adapter artifact is read,
    // so it is sized for the largest registered rank; the bank's actual rank lives on the model
    // view and is what the Op executes. The difference is well under a MiB of transient scratch.
    std::uint32_t lora_sizing_rank = 0;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] bool lora() const noexcept { return lora_slots > 0; }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    // The workspace layout is frozen here, before the pool is scanned, so the slot count is the
    // requested one rather than the one the bank settles on after clamping to the pool size. A
    // slot the pool cannot fill costs transient scratch, not device residency.
    const bool enabled = !options.lora.directory.empty() && options.lora.slots > 0;
    return StartupFeatures{
        .vision            = options.enable_vision,
        .vision_max_tokens = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .speculative       = options.speculative.backend,
        .proposal_head     = options.speculative.proposal_head,
        .lora_slots        = enabled ? options.lora.slots : 0U,
        .lora_sizing_rank  = enabled ? ops::kMaximumLoraRank : 0U,
    };
}

} // namespace ninfer::targets::qwen3_8

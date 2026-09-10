#pragma once

#include "runtime/contract/types.h"

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Detects a locked repeating cycle in a request's committed token stream.
//
// A recurring n-gram proposes a period: the distance back to its previous occurrence. The period is
// then held and extended by comparing each new token against the token one period earlier, so the
// run of confirmed tokens is exact and never depends on the proposal structure surviving. The guard
// trips once that run covers `cycles` whole periods and at least `min_tokens` tokens, the second
// condition being what stops a short period such as a divider rule from tripping on a few tokens.
//
// Only committed tokens are observed. Speculative drafts that were proposed and rejected never
// reach it, and it reads no logits, so it cannot perturb sampling: a generation that does not lock
// into a cycle is bit-identical to one produced with the guard absent.
class RepetitionGuard {
public:
    explicit RepetitionGuard(const RepetitionGuardOptions& options)
        : window_(options.window), ngram_(options.ngram), cycles_(options.cycles),
          min_tokens_(options.min_tokens) {
        if (window_ == 0 || ngram_ == 0 || cycles_ == 0) {
            throw std::invalid_argument(
                "repetition guard window, ngram and cycles must be positive");
        }
        if (ngram_ > window_) {
            throw std::invalid_argument("repetition guard ngram must fit inside its window");
        }
        history_.assign(window_, 0);
        proposals_.assign(proposal_capacity(window_), Proposal{});
        mask_ = static_cast<std::uint32_t>(proposals_.size() - 1);
        for (std::uint32_t index = 1; index < ngram_; ++index) { drop_factor_ *= kHashBase; }
    }

    // Folds committed tokens in stream order. Returns true once the cycle is locked; the guard
    // latches, so every later call returns true without further work.
    bool observe(std::span<const TokenId> tokens) noexcept {
        for (const TokenId token : tokens) {
            if (tripped_) { return true; }
            admit(token);
        }
        return tripped_;
    }

    [[nodiscard]] bool tripped() const noexcept { return tripped_; }

    // Length of the confirmed cycle, valid once tripped, so a caller can report why a generation
    // was cut without re-deriving it.
    [[nodiscard]] std::uint32_t period() const noexcept { return period_; }

private:
    // A truncated hash is enough because a proposal is only a hint: the period it suggests is
    // confirmed token by token against history before it can contribute to a trip.
    struct Proposal {
        std::uint32_t hash     = 0;
        std::uint64_t position = 0;
    };

    static constexpr std::uint64_t kHashBase = 0x100000001B3ULL;

    // Two entries per window position, direct-mapped. A generation runs far past the window, so an
    // open-addressed table would fill and never free a slot; a colliding write simply evicts the
    // older n-gram, costing at most a later proposal.
    static std::size_t proposal_capacity(std::uint32_t window) noexcept {
        std::size_t capacity = 1;
        while (capacity < static_cast<std::size_t>(window) * 2) { capacity <<= 1U; }
        return capacity;
    }

    [[nodiscard]] std::uint64_t token_at(std::uint64_t position) const noexcept {
        return history_[static_cast<std::size_t>(position % window_)];
    }

    void admit(TokenId token) noexcept {
        const std::uint64_t entering =
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(token));
        const std::uint64_t leaving = position_ >= ngram_ ? token_at(position_ - ngram_) : 0;
        history_[static_cast<std::size_t>(position_ % window_)] = entering;
        ++position_;

        hash_ = hash_ * kHashBase + entering;
        if (position_ > ngram_) { hash_ -= leaving * drop_factor_ * kHashBase; }
        if (position_ < ngram_) { return; }

        if (period_ != 0 && extend(entering)) { return; }
        propose();
    }

    // Holds the active period for as long as the stream keeps agreeing with itself one period back.
    bool extend(std::uint64_t entering) noexcept {
        if (entering != token_at(position_ - 1 - period_)) {
            period_ = 0;
            run_    = 0;
            return false;
        }
        ++run_;
        if (run_ >= static_cast<std::uint64_t>(cycles_) * period_ && run_ >= min_tokens_) {
            tripped_ = true;
        }
        return true;
    }

    // Records this n-gram and adopts the distance to its previous occurrence as the next candidate
    // period. The matching n-gram itself is that many already-agreeing tokens.
    void propose() noexcept {
        const auto folded       = static_cast<std::uint32_t>(hash_ ^ (hash_ >> 32U));
        Proposal& proposal      = proposals_[static_cast<std::size_t>(folded & mask_)];
        const std::uint64_t gap = position_ - proposal.position;
        if (proposal.position != 0 && proposal.hash == folded && gap <= window_ - ngram_) {
            period_ = static_cast<std::uint32_t>(gap);
            run_    = ngram_;
        }
        proposal.hash     = folded;
        proposal.position = position_;
    }

    std::uint32_t window_     = 0;
    std::uint32_t ngram_      = 0;
    std::uint32_t cycles_     = 0;
    std::uint32_t min_tokens_ = 0;

    std::vector<std::uint64_t> history_;
    std::vector<Proposal> proposals_;
    std::uint32_t mask_        = 0;
    std::uint64_t drop_factor_ = 1;
    std::uint64_t hash_        = 0;
    std::uint64_t position_    = 0;
    std::uint64_t run_         = 0;
    std::uint32_t period_      = 0;
    bool tripped_              = false;
};

} // namespace ninfer::runtime

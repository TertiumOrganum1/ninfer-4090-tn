#pragma once

#include <cstdint>

namespace ninfer::runtime {

// The contract every resolved decode row must satisfy before the target folds it into sequence
// state. It lives here rather than inside the invariant that enforces it so the executor and the
// target agree on one rule, and so the rule can be tested without a device.
//
// Only a cancelled row may commit nothing: its tokens are discarded and its lane is aborted, so
// there is no prefix to fold. Every other row ends a round that really produced tokens, and a
// terminal row is the only one allowed to keep less than all of them -- that is how a stop token
// or a between-round decision cuts a speculative span short. A non-terminal row that dropped
// tokens would leave the sequence and its KV disagreeing about where generation reached.
[[nodiscard]] constexpr bool row_commit_is_licensed(bool cancelled, std::uint32_t accepted,
                                                    std::uint32_t produced,
                                                    bool terminal) noexcept {
    if (cancelled) { return accepted == 0; }
    return accepted != 0 && accepted <= produced && (terminal || accepted == produced);
}

} // namespace ninfer::runtime

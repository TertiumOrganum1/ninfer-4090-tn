#include "runtime/generation/row_commit.h"

#include <iostream>

namespace {

using ninfer::runtime::row_commit_is_licensed;

constexpr bool kCancelled    = true;
constexpr bool kLive         = false;
constexpr bool kTerminal     = true;
constexpr bool kContinuing   = false;
constexpr std::uint32_t kRun = 4;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int cancelled_rows_commit_nothing() {
    return check(row_commit_is_licensed(kCancelled, 0, kRun, kTerminal),
                 "a cancelled row committing nothing must be licensed") +
           check(!row_commit_is_licensed(kCancelled, 1, kRun, kTerminal),
                 "a cancelled row must not commit any of its tokens");
}

// The shape that bricked the engine: a between-round terminal decision accepts no tokens, which
// is licensed only when the row is also cancelled. The repetition guard once resolved its rows
// this way, and the target answered by retiring the whole executor.
int a_live_row_never_commits_nothing() {
    return check(!row_commit_is_licensed(kLive, 0, kRun, kTerminal),
                 "a terminal live row committing nothing must be refused") +
           check(!row_commit_is_licensed(kLive, 0, kRun, kContinuing),
                 "a continuing live row committing nothing must be refused");
}

int a_terminal_row_may_keep_part_of_its_span() {
    return check(row_commit_is_licensed(kLive, 1, kRun, kTerminal),
                 "a terminal row must be free to cut its span short") +
           check(row_commit_is_licensed(kLive, kRun, kRun, kTerminal),
                 "a terminal row must be free to keep its whole span");
}

int a_continuing_row_keeps_its_whole_span() {
    return check(row_commit_is_licensed(kLive, kRun, kRun, kContinuing),
                 "a continuing row keeping everything must be licensed") +
           check(!row_commit_is_licensed(kLive, kRun - 1, kRun, kContinuing),
                 "a continuing row dropping tokens must be refused");
}

int no_row_commits_more_than_it_produced() {
    return check(!row_commit_is_licensed(kLive, kRun + 1, kRun, kTerminal),
                 "a terminal row must not commit past its span") +
           check(!row_commit_is_licensed(kLive, kRun + 1, kRun, kContinuing),
                 "a continuing row must not commit past its span");
}

} // namespace

int main() {
    const int failures = cancelled_rows_commit_nothing() + a_live_row_never_commits_nothing() +
                         a_terminal_row_may_keep_part_of_its_span() +
                         a_continuing_row_keeps_its_whole_span() +
                         no_row_commits_more_than_it_produced();
    if (failures != 0) {
        std::cerr << failures << " row commit checks failed\n";
        return 1;
    }
    return 0;
}

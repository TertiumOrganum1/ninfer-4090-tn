#include "runtime/generation/repetition_guard.h"

#include <iostream>
#include <numeric>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::RepetitionGuardOptions;
using ninfer::runtime::RepetitionGuard;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<TokenId> phrase(TokenId first, std::uint32_t length) {
    std::vector<TokenId> tokens(length);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

void extend(std::vector<TokenId>& stream, const std::vector<TokenId>& tail) {
    stream.insert(stream.end(), tail.begin(), tail.end());
}

// Feeds the stream in four-token rounds, the shape a draft window of three produces, so the guard
// is exercised the way the decode loop calls it rather than as one flat span.
bool observe_in_rounds(RepetitionGuard& guard, const std::vector<TokenId>& stream) {
    constexpr std::size_t kRound = 4;
    for (std::size_t offset = 0; offset < stream.size(); offset += kRound) {
        const std::size_t count = std::min(kRound, stream.size() - offset);
        if (guard.observe(std::span<const TokenId>(stream.data() + offset, count))) { return true; }
    }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const RepetitionGuardOptions defaults{};

    // The observed failure: a model that could not emit a tool call narrates its next step, reads
    // the narration back and alternates between two spellings of it forever. No line repeats
    // consecutively, so only cycle detection catches this.
    {
        const std::vector<TokenId> let_me = phrase(1000, 14);
        const std::vector<TokenId> i_will = phrase(2000, 14);
        std::vector<TokenId> stream       = phrase(10, 200);
        for (int cycle = 0; cycle < 12; ++cycle) {
            extend(stream, let_me);
            extend(stream, i_will);
        }
        RepetitionGuard guard(defaults);
        failures +=
            check(observe_in_rounds(guard, stream), "an alternating two-phrase cycle ran on");
        failures += check(guard.period() == 28, "the confirmed period was not the two-phrase unit");
    }

    // Prose that never repeats must run to its natural end.
    {
        RepetitionGuard guard(defaults);
        const std::vector<TokenId> stream = phrase(1, 4000);
        failures += check(!observe_in_rounds(guard, stream), "unique output was cut as a cycle");
    }

    // A divider rule is one token repeated far more often than any n-gram needs, and is the case a
    // bare period-1 match would misread.
    {
        RepetitionGuard guard(defaults);
        std::vector<TokenId> stream = phrase(1, 100);
        stream.insert(stream.end(), 40, TokenId{61});
        extend(stream, phrase(500, 100));
        failures += check(!observe_in_rounds(guard, stream), "a divider rule was read as a cycle");
    }

    // Structurally similar lines whose contents differ are the common legitimate repetition: a
    // table, a field list, an import block. The shared prefix must not lock a period.
    {
        RepetitionGuard guard(defaults);
        std::vector<TokenId> stream;
        for (TokenId row = 0; row < 60; ++row) {
            extend(stream, phrase(700, 6));
            extend(stream, phrase(5000 + row * 8, 8));
        }
        failures +=
            check(!observe_in_rounds(guard, stream), "a repeating row shape was read as a cycle");
    }

    // Latching: once the cycle is confirmed the guard keeps reporting it, so a caller that asks
    // again on a later round still terminates.
    {
        RepetitionGuard guard(defaults);
        std::vector<TokenId> stream = phrase(10, 50);
        for (int cycle = 0; cycle < 20; ++cycle) { extend(stream, phrase(3000, 30)); }
        failures += check(observe_in_rounds(guard, stream), "a single repeated phrase ran on");
        const std::vector<TokenId> unrelated = phrase(9000, 4);
        failures += check(guard.observe(unrelated), "the guard unlatched after tripping");
    }

    // Disabling is the parity escape hatch, and it is expressed by not constructing a guard at all.
    // What must hold here is that a window smaller than the cycle cannot see it.
    {
        RepetitionGuardOptions narrow = defaults;
        narrow.window                 = 16;
        narrow.ngram                  = 8;
        RepetitionGuard guard(narrow);
        std::vector<TokenId> stream;
        for (int cycle = 0; cycle < 30; ++cycle) { extend(stream, phrase(4000, 40)); }
        failures += check(!observe_in_rounds(guard, stream),
                          "a period longer than the window was reported");
    }

    if (failures == 0) { std::cout << "repetition guard ok\n"; }
    return failures == 0 ? 0 : 1;
}

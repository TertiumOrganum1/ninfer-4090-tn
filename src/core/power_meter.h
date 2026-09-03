#pragma once

// Board power and energy, read from NVML.
//
// NVML ships with the driver this engine already requires, so it is linked directly rather than
// probed. Boards that mask the management interface, or that do not implement a given counter,
// degrade to `available() == false` / `energy_available() == false`; every derived metric then
// renders as absent rather than as zero, because a reading that was never taken is not a zero.
//
// The two readings have very different costs and resolutions, and callers must choose deliberately.
// Measured on an RTX 4090 (Ada, driver 610.57.04):
//
//   energy_joules()   Exact, integrated by the board itself, so it carries no sampling error at
//                     all. But it costs ~2.9 ms per call and advances in ~100 ms steps. Differenced
//                     over 5 s it reproduces integrated instantaneous power to within noise; over
//                     0.5 s it is off by ~25%. Sample it from an interval-boundary thread only,
//                     never from an execution loop.
//   instant_watts()   ~1.4 us, cheap enough to sample at execution-unit boundaries. It is a point
//                     sample of a quantity the board refreshes at ~50 Hz, so one reading does not
//                     characterize one decode round. Only its time integral over many units
//                     converges, and that integral is what gets reconciled against energy_joules().
//
// nvmlDeviceGetPowerUsage is deliberately not used. On Ampere and newer it returns power averaged
// over a trailing 1 s window, which lags a prefill/decode transition by far more than the phase it
// is meant to attribute.

#include <optional>
#include <string>

namespace ninfer::core {

// Owns one NVML session and a pinned device handle. Construction performs the whole handshake and
// probes which counters the board implements, so the read paths cost only their driver call.
class PowerMeter {
public:
    explicit PowerMeter(int device);
    ~PowerMeter();

    PowerMeter(const PowerMeter&)            = delete;
    PowerMeter& operator=(const PowerMeter&) = delete;

    [[nodiscard]] bool available() const noexcept { return initialized_; }

    // Set only when available() is false, or when a probed counter is missing.
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

    // Whether this board implements the cumulative energy counter. Probed once: it is a fixed
    // capability of the part and driver, not a transient condition. Many GeForce boards lack it.
    [[nodiscard]] bool energy_available() const noexcept { return energy_available_; }

    // Board energy since driver load, in joules. Monotonic within a driver session and reset by a
    // driver reload, so consumers must difference it defensively rather than trusting ordering.
    [[nodiscard]] std::optional<double> energy_joules() const;

    // Instantaneous board power, in watts.
    [[nodiscard]] std::optional<double> instant_watts() const;

    // Board power averaged by the driver over a trailing 1 s window, in watts. This is the better
    // statistic for a once-per-second display gauge, and the worse one for attributing a phase
    // that lasts tens of milliseconds; the two readings are exposed separately rather than one
    // being made to serve both. Nullopt on parts that do not implement the averaged field.
    [[nodiscard]] std::optional<double> average_watts() const;

    // Enforced power ceiling in watts, zero where the board reports none. Energy efficiency is a
    // function of this ceiling, so any published joules-per-token figure is only meaningful
    // alongside it.
    [[nodiscard]] double power_limit_watts() const noexcept { return power_limit_watts_; }

private:
    [[nodiscard]] std::optional<double> read_power_field(unsigned int field_id) const;

    int device_               = 0;
    bool initialized_         = false;
    bool energy_available_    = false;
    double power_limit_watts_ = 0.0;
    std::string error_;
    void* handle_ = nullptr; // nvmlDevice_t, kept opaque so nvml.h stays out of this header
};

} // namespace ninfer::core

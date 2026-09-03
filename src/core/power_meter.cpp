#include "core/power_meter.h"

#include <nvml.h>

namespace ninfer::core {

PowerMeter::PowerMeter(int device) : device_(device) {
    nvmlReturn_t status = nvmlInit_v2();
    if (status != NVML_SUCCESS) {
        error_ = std::string("nvmlInit_v2: ") + nvmlErrorString(status);
        return;
    }

    nvmlDevice_t handle = nullptr;
    status = nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_), &handle);
    if (status != NVML_SUCCESS) {
        error_ = std::string("nvmlDeviceGetHandleByIndex_v2: ") + nvmlErrorString(status);
        nvmlShutdown();
        return;
    }

    handle_      = handle;
    initialized_ = true;

    unsigned int power_limit_mw = 0;
    if (nvmlDeviceGetEnforcedPowerLimit(handle, &power_limit_mw) == NVML_SUCCESS) {
        power_limit_watts_ = static_cast<double>(power_limit_mw) / 1000.0;
    }

    // Probe the energy counter once. It is the only exact energy source available, so whether the
    // board has it decides between measured and estimated aggregates downstream.
    unsigned long long energy_mj = 0;
    status                       = nvmlDeviceGetTotalEnergyConsumption(handle, &energy_mj);
    if (status == NVML_SUCCESS) {
        energy_available_ = true;
    } else {
        error_ = std::string("nvmlDeviceGetTotalEnergyConsumption: ") + nvmlErrorString(status);
    }
}

PowerMeter::~PowerMeter() {
    if (initialized_) { nvmlShutdown(); }
}

std::optional<double> PowerMeter::energy_joules() const {
    if (!initialized_ || !energy_available_) { return std::nullopt; }
    unsigned long long energy_mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(static_cast<nvmlDevice_t>(handle_), &energy_mj) !=
        NVML_SUCCESS) {
        return std::nullopt;
    }
    return static_cast<double>(energy_mj) / 1000.0;
}

std::optional<double> PowerMeter::read_power_field(unsigned int field_id) const {
    if (!initialized_) { return std::nullopt; }
    nvmlFieldValue_t field{};
    field.fieldId = field_id;
    field.scopeId = 0; // GPU scope; module scope exists only on parts that pair a CPU with the GPU
    if (nvmlDeviceGetFieldValues(static_cast<nvmlDevice_t>(handle_), 1, &field) != NVML_SUCCESS) {
        return std::nullopt;
    }
    if (field.nvmlReturn != NVML_SUCCESS || field.valueType != NVML_VALUE_TYPE_UNSIGNED_INT) {
        return std::nullopt;
    }
    return static_cast<double>(field.value.uiVal) / 1000.0;
}

std::optional<double> PowerMeter::instant_watts() const {
    return read_power_field(NVML_FI_DEV_POWER_INSTANT);
}

std::optional<double> PowerMeter::average_watts() const {
    return read_power_field(NVML_FI_DEV_POWER_AVERAGE);
}

} // namespace ninfer::core

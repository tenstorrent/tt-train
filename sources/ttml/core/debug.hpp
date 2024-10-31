#pragma once

namespace ttml::core::debug {

struct Debug {
    static constexpr bool enable_backward_performance_measurement = false;
    static constexpr bool enable_print_tensor_stats = false;
};

}  // namespace ttml::core::debug

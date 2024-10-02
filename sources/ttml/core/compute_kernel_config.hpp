#pragma once

#include "ttnn_all_includes.hpp"

namespace ttml::core {

class ComputeKernelConfig {
public:
    static ttnn::WormholeComputeKernelConfig precise();
    static ttnn::WormholeComputeKernelConfig fast();
};

}  // namespace ttml::core
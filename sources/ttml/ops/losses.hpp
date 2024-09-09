#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

enum ReduceType : uint8_t { MEAN = 0, SUM = 1 };

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction, ReduceType reduce = ReduceType::MEAN);

}  // namespace ttml::ops
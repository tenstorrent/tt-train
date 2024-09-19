#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr layernorm(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& gamma, const autograd::TensorPtr& beta);

}  // namespace ttml::ops
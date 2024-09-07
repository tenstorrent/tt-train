#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor);
}  // namespace ttml::ops
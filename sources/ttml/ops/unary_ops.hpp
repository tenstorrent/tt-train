#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);
autograd::TensorPtr identity(const autograd::TensorPtr& tensor);
autograd::TensorPtr gelu(const autograd::TensorPtr& tensor);
autograd::TensorPtr mean(const autograd::TensorPtr& tensor);
autograd::TensorPtr sum(const autograd::TensorPtr& tensor);
autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim);
}  // namespace ttml::ops
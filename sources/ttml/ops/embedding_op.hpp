#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr embedding_op(const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight);

}  // namespace ttml::ops
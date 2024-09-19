#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability);

}  // namespace ttml::ops
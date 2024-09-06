#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor);

}  // namespace ttml::ops
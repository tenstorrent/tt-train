#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {
autograd::TensorPtr layernorm(const autograd::TensorPtr& tensor);
}
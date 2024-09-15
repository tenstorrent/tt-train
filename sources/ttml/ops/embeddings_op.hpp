#pragma once
#include "autograd/tensor.hpp"

namespace ttml::ops {
autograd::TensorPtr embedding(const autograd::TensorPtr& tensor);
}
#pragma once
#include "autograd/tensor.hpp"
namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
autograd::TensorPtr operator*(const autograd::TensorPtr& a, const autograd::TensorPtr& b);

}  // namespace ttml::ops
#pragma once
#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b);
}
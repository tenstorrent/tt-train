#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr scaled_dot_product_attention(
    autograd::TensorPtr query,
    autograd::TensorPtr key,
    autograd::TensorPtr value,
    std::optional<autograd::TensorPtr> mask = std::nullopt);

}
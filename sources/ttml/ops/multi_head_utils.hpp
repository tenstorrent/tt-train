
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr heads_creation(autograd::TensorPtr x, uint32_t num_heads);

autograd::TensorPtr heads_fusion(autograd::TensorPtr x);

}  // namespace ttml::ops
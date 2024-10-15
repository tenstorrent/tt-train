// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr heads_creation(const autograd::TensorPtr& x, uint32_t num_heads);

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x);

}  // namespace ttml::ops
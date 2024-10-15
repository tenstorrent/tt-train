// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ttnn/tensor/tensor.hpp>

#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim);
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor max(const tt::tt_metal::Tensor& t, int dim, bool keepdim = true);
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim);
tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b);

}  // namespace ttml::ttnn_fixed

// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <ttnn/operations/moreh/moreh_sum/moreh_sum.hpp>

#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim) {
    return ttnn::moreh_sum(
        t,
        /* dim */ dim,
        /* keep_dim */ true,
        /* output */ std::nullopt,
        /* output_mem_config */ std::nullopt,
        /*compute_kernel_config */ core::ComputeKernelConfig::precise());
}

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    return sum_over_dim(t, /* dim */ 0);
}

// This is a workaround for the lack of working `ttnn::max` implementation.
tt::tt_metal::Tensor max(const tt::tt_metal::Tensor& t, int dim, bool keepdim) {
    return ttnn::max(t, dim, keepdim);
}

// Stable softmax implementation
// ttnn::softmax also exists, but it is not stable (even after max subtraction optimization)
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto t_max = ttnn_fixed::max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, t_max);
    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim = sum_over_dim(t_sub_max_exp, dim);
    auto inv_t_sum_over_dim = ttnn::reciprocal(/* queue_id */ 0, t_sum_over_dim);
    return ttnn::multiply(t_sub_max_exp, inv_t_sum_over_dim);
}

tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    auto inv_b = ttnn::reciprocal(/* queue_id */ 0, b);
    return ttnn::multiply(a, inv_b);
}

}  // namespace ttml::ttnn_fixed

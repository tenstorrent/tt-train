// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/clip_gradient_norm.hpp"

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

void clip_tensor_norm_(tt::tt_metal::Tensor& tensor, float max_norm) {
    if (max_norm <= 0.F) {
        throw std::logic_error(fmt::format("max_norm should be positive, current max norm {}", max_norm));
    }

    auto squared = ttnn::multiply(tensor, tensor);
    auto shape = core::create_shape({1, 1, 1, 1});
    auto out = ttml::core::from_vector({0.F}, shape, &ttml::autograd::ctx().get_device());
    ttnn::moreh_sum(squared, std::nullopt, true, out, squared.memory_config(), core::ComputeKernelConfig::precise());
    auto grad_norm_tensor = ttnn::sqrt(out);

    auto grad_norm_tensor_repeated = ttnn::repeat(grad_norm_tensor, tensor.get_logical_shape());
    auto inv_grad_norm_tensor = ttnn::reciprocal(grad_norm_tensor);
    auto scale = ttnn::multiply(inv_grad_norm_tensor, max_norm);
    auto scaled_tensor = ttnn::multiply(tensor, scale);
    core::print_tensor_stats(tensor, "tensor");
    core::print_tensor_stats(grad_norm_tensor, "grad_norm_tensor");
    core::print_tensor_stats(grad_norm_tensor_repeated, "grad_norm_tensor_repeated");
    tensor = ttnn::where(ttnn::gt(grad_norm_tensor_repeated, max_norm), scaled_tensor, tensor);
}

}  // namespace ttml::autograd

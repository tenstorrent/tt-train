// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr rmsnorm_op(const autograd::TensorPtr& input, const autograd::TensorPtr& weight) {
    const float eps = 1e-8F;
    const auto& input_tensor = input->get_value();
    const auto& weight_tensor = weight->get_value();

    auto normalized_tensor = ttnn::rms_norm(
        input_tensor,
        /* epsilon */ eps,
        /* weight */ std::nullopt,
        /* bias */ std::nullopt,
        /* residual_input_tensor */ std::nullopt,
        /* memory_config */ std::nullopt,
        /* program_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());

    auto result = ttnn::multiply(normalized_tensor, weight_tensor);
    auto out = autograd::create_tensor(result);

    autograd::GradFunction grad = [input, weight, out, normalized_tensor]() {

    };

    auto links = autograd::get_links(input, weight);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops

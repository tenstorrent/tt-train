// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_op.hpp"

#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
namespace ttml::ops {

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::linear(
        tensor->get_value(),
        weight->get_value(),
        bias->get_value(),
        /* transpose_a */ false,
        /* tranpose_b */ true,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::fast()));

    autograd::GradFunction grad = [weight, bias, tensor, out]() {
        auto bias_grad = core::zeros_like(bias->get_value());
        auto tensor_grad = core::zeros_like(tensor->get_value());
        auto weight_grad = core::zeros_like(weight->get_value());

        auto res = ttnn::moreh_linear_backward(
            out->get_grad(),
            tensor->get_value(),
            weight->get_value(),
            /* are required outputs */ std::vector<bool>{true, true, true},
            bias->get_value(),
            tensor_grad,
            weight_grad,
            bias_grad,
            /* input_grad_mem_config */ std::nullopt,
            /* weight_grad_mem_config */ std::nullopt,
            /* bias_grad_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::fast());

        tensor->add_grad(res[0].value());
        weight->add_grad(res[1].value());
        bias->add_grad(res[2].value());
    };

    std::vector<autograd::NodeId> links = autograd::get_links(weight, tensor, bias);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops

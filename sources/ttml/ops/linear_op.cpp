// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */
        config,
        /* core_grid */ ttnn::CoreGrid{7, 8},
        /* output_tile */ std::nullopt);
}

}  // namespace

namespace ttml::ops {

void ttnn_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out,
    const ttnn::WormholeComputeKernelConfig& config) {
    const auto& tensor_value = tensor->get_value();
    auto volume_without_features = tensor_value.get_logical_volume() / tensor_value.get_shape()[-1];
    auto reshaped_tensor =
        ttnn::reshape(tensor_value, ttnn::Shape({volume_without_features, tensor_value.get_shape()[-1]}));

    auto reshaped_grad =
        ttnn::reshape(out->get_grad(), ttnn::Shape({volume_without_features, out->get_grad().get_shape()[-1]}));
    auto reshaped_bias_grad = ttnn_fixed::sum_over_dim(reshaped_grad, /* axis */ 0);
    auto reshaped_weight_grad =
        matmul(reshaped_grad, reshaped_tensor, /* transpose_a */ true, /* transpose_b */ false, config);
    auto reshaped_tensor_grad =
        matmul(reshaped_grad, weight->get_value(), /* transpose_a */ false, /* transpose_b */ false, config);

    auto bias_grad = ttnn::reshape(reshaped_bias_grad, bias->get_value().get_shape());
    auto weight_grad = ttnn::reshape(reshaped_weight_grad, weight->get_value().get_shape());
    auto tensor_grad = ttnn::reshape(reshaped_tensor_grad, tensor_value.get_shape());

    tensor->add_grad(tensor_grad);
    weight->add_grad(weight_grad);
    bias->add_grad(bias_grad);
}

void moreh_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out,
    const ttnn::WormholeComputeKernelConfig& config) {
    auto bias_grad = ttnn::empty_like(bias->get_value());
    auto tensor_grad = ttnn::empty_like(tensor->get_value());
    auto weight_grad = ttnn::empty_like(weight->get_value());

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
        /* compute_kernel_config */ config);

    if (!res[0].has_value()) {
        throw std::runtime_error("Tensor gradient is not available");
    }
    tensor->add_grad(res[0].value());

    if (!res[1].has_value()) {
        throw std::runtime_error("Weight gradient is not available");
    }
    weight->add_grad(res[1].value());

    if (!res[2].has_value()) {
        throw std::runtime_error("Bias gradient is not available");
    }
    bias->add_grad(res[2].value());
}

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
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
        /* core_grid */ ttnn::CoreGrid{7, 8}));

    autograd::GradFunction grad = [weight, bias, tensor, out]() {
        auto tensor_shape = tensor->get_value().get_shape();
        auto grad_shape = out->get_grad().get_shape();
        // for some reason, reshape produces wrong values when last dimensions not divisible by TILE
        if (tensor_shape[-2] % TILE_HEIGHT != 0 ||
            tensor_shape[-1] % TILE_WIDTH != 0 && grad_shape[-1] % TILE_WIDTH != 0) {
            moreh_linear_backward(tensor, weight, bias, out);
        } else {
            ttnn_linear_backward(tensor, weight, bias, out);
        }
    };

    auto links = autograd::get_links(weight, tensor, bias);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops

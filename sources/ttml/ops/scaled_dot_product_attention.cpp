// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scaled_dot_product_attention.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    bool use_ttnn = false) {
    if (use_ttnn) {
        return ttnn::matmul(
            a,
            b,
            /* transpose_a */ transpose_a,
            /* tranpose_b */ transpose_b,
            /* memory_config */ std::nullopt,
            /* dtype */ std::nullopt,
            /* program_config */ std::nullopt,
            /* activation */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::fast());
    }

    return ttnn::moreh_matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* output */ std::nullopt,
        /* bias */ std::nullopt,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::fast());
}

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto qk_t = matmul(query->get_value(), key->get_value(), /* transpose_a */ false, /* transpose_b */ true);
    auto attention_weights = ttnn::scale_mask_softmax_in_place(
        qk_t,
        scale,
        mask.value()->get_value(),
        ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
        /* is_causal_mask */ true,
        ttml::core::ComputeKernelConfig::softmax(),
        /* numeric_stable */ true);
    // TODO: add dropout here

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv =
        matmul(attention_weights, value->get_value(), /* transpose_a */ false, /* transpose_b */ false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad = [scale, query, key, value, attention_weights, out, mask]() {
        auto grad_output = out->get_grad();
        // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
        auto grad_v = matmul(attention_weights, grad_output, /* transpose_a */ true, /* transpose_b */ false);
        auto grad_attention_weights =
            matmul(grad_output, value->get_value(), /* transpose_a */ false, /* transpose_b */ true);
        auto grad_scaled_dot = ttnn::multiply(
            attention_weights,
            ttnn::subtract(
                grad_attention_weights,
                ttnn_fixed::sum_over_dim(ttnn::multiply(attention_weights, grad_attention_weights), 3)));
        if (mask.has_value()) {
            grad_scaled_dot = ttnn::multiply(grad_scaled_dot, mask.value()->get_value());
        }
        auto grad_q = matmul(
            grad_scaled_dot,
            key->get_value(),
            /* transpose_a */ false,
            /* transpose_b */ false);
        grad_q = ttnn::multiply(grad_q, scale);

        auto grad_k = matmul(
            grad_scaled_dot,
            query->get_value(),
            /* transpose_a */ true,
            /* transpose_b */ false);
        grad_k = ttnn::multiply(grad_k, scale);

        query->add_grad(grad_q);
        key->add_grad(grad_k);
        value->add_grad(grad_v);
    };

    auto links = autograd::get_links(query, key, value);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr scaled_sigmoid_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_shape()[-1]));
    // (B, H, S, E) x (B, H, E, S) -> (B, H, S, S)
    auto qk_t = matmul(query->get_value(), key->get_value(), /* transpose_a */ false, /* transpose_b */ true);
    // (B, H, S, S) * scale
    auto qk_scaled = ttnn::multiply(qk_t, scale);
    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value()->get_value(), qk_scaled, /* other */ -1e9F);
    }
    // (B, H, S, S)
    // auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    auto attention_weights =
        ttnn::sigmoid(ttnn::subtract(qk_scaled, std::logf(static_cast<float>(query->get_value().get_shape()[-2]))));

    // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
    auto attention_qkv =
        matmul(attention_weights, value->get_value(), /* transpose_a */ false, /* transpose_b */ false);
    auto out = ttml::autograd::create_tensor(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, qk_t, qk_scaled, attention_weights, attention_qkv, out, mask]() {
            auto grad_output = out->get_grad();
            // (B, H, S, S) x (B, H, S, E) -> (B, H, S, E)
            auto grad_v = matmul(attention_weights, grad_output, /* transpose_a */ true, /* transpose_b */ false);
            auto grad_attention_weights =
                matmul(grad_output, value->get_value(), /* transpose_a */ false, /* transpose_b */ true);
            auto grad_scaled_dot =
                ttnn::sigmoid_bw(
                    grad_attention_weights,
                    ttnn::subtract(qk_scaled, std::logf(static_cast<float>(query->get_value().get_shape()[-2]))))
                    .front();

            if (mask.has_value()) {
                grad_scaled_dot = ttnn::where(mask.value()->get_value(), grad_scaled_dot, /* other */ 0.0F);
            }

            auto grad_q = matmul(
                grad_scaled_dot,
                key->get_value(),
                /* transpose_a */ false,
                /* transpose_b */ false);
            grad_q = ttnn::multiply(grad_q, scale);

            auto grad_k = matmul(
                grad_scaled_dot,
                query->get_value(),
                /* transpose_a */ true,
                /* transpose_b */ false);
            grad_k = ttnn::multiply(grad_k, scale);

            query->add_grad(grad_q);
            key->add_grad(grad_k);
            value->add_grad(grad_v);
        };

    auto links = autograd::get_links(query, key, value);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops

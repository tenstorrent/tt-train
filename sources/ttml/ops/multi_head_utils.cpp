// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_utils.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> heads_creation(
    const autograd::TensorPtr& query, const autograd::TensorPtr& key_value, uint32_t num_heads) {
    auto [q, k, v] = ttnn::experimental::nlp_create_qkv_heads(
        query->get_value(),
        key_value->get_value(),
        num_heads,
        num_heads,
        /* transpose_k */ false,
        /* memory_config */ std::nullopt,
        /* optional_output_tensors */ std::nullopt);

    auto out_q = autograd::create_tensor(q);
    auto out_k = autograd::create_tensor(k);
    auto out_v = autograd::create_tensor(v);

    autograd::GradFunction grad_q = [out_q, query]() {
        auto grad = out_q->get_grad();
        auto result = ttnn::experimental::nlp_concat_heads(grad);
        query->add_grad(result);
    };

    autograd::GradFunction grad_kv = [out_k, out_v, key_value]() {
        auto grad_k = out_k->get_grad();
        auto grad_v = out_v->get_grad();
        auto result = ttnn::concat(std::vector<ttnn::Tensor>({grad_k, grad_v}), /* dim */ 3);
        result = ttnn::experimental::nlp_concat_heads(result);
        key_value->add_grad(result);
    };

    auto links_q = autograd::get_links(query);
    out_q->set_node(autograd::ctx().add_backward_node(std::move(grad_q), links_q));
    auto links_k = autograd::get_links(key_value);
    out_k->set_node(autograd::ctx().add_backward_node(std::move(grad_kv), links_k));
    auto links_v = autograd::get_links(key_value, out_k);
    out_v->set_node(autograd::ctx().add_backward_node([]() {}, links_v));
    return {out_q, out_k, out_v};
}

autograd::TensorPtr heads_fusion(const autograd::TensorPtr& x) {
    auto x_shape = x->get_value().get_shape();

    uint32_t batch_size = x_shape[0];
    uint32_t num_heads = x_shape[1];
    uint32_t sequence_length = x_shape[2];
    uint32_t embedding_dim = x_shape[3];

    auto fused_heads = ttnn::experimental::nlp_concat_heads(x->get_value());
    auto out = autograd::create_tensor(fused_heads);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim]() {
        auto grad_output = out->get_grad();
        // (B, 1, S, E) -> (B, 1, E, S)
        auto grad_result = ttnn::transpose(grad_output, -2, -1);
        // (B, 1, E, S) -> (B, H, E/H, S)
        grad_result =
            ttnn::reshape(grad_result, core::create_shape({batch_size, num_heads, embedding_dim, sequence_length}));
        // (B, H, E/H, S) -> (B, H, S, E/H)
        grad_result = ttnn::transpose(grad_result, -2, -1);
        x->add_grad(grad_result);
    };

    auto links = autograd::get_links(x);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops

#include "scaled_dot_product_attention.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr scaled_dot_product_attention(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask) {
    const float scale = 1.0F / std::sqrtf(static_cast<float>(query->get_value().get_shape()[-1]));
    auto qk_t = ttnn::matmul(query->get_value(), key->get_value(), /* transpose_a */ false, /* transpose_b */ true);
    auto qk_scaled = ttnn::multiply(qk_t, scale);
    if (mask.has_value()) {
        qk_scaled = ttnn::where(mask.value()->get_value(), qk_scaled, /* other */ -1e9F);
    }
    auto attention_weights = ttnn_fixed::softmax(qk_scaled, /* axis */ 3);
    // TODO: add dropout here
    auto attention_qkv =
        ttnn::matmul(attention_weights, value->get_value(), /* transpose_a */ false, /* transpose_b */ false);

    auto out = ttml::autograd::create_tensor();
    out->set_value(attention_qkv);

    ttml::autograd::GradFunction grad =
        [scale, query, key, value, qk_t, qk_scaled, attention_weights, attention_qkv, out, mask]() {
            auto grad_output = out->get_grad();
            auto grad_v = ttnn::matmul(attention_weights, grad_output, /* transpose_a */ true, /* transpose_b */ false);
            auto grad_attention_weights =
                ttnn::matmul(grad_output, value->get_value(), /* transpose_a */ false, /* transpose_b */ true);
            auto grad_scaled_dot = ttnn::multiply(
                attention_weights,
                ttnn::subtract(
                    grad_attention_weights, ttnn::sum(ttnn::multiply(attention_weights, grad_attention_weights), 3)));
            if (mask.has_value()) {
                grad_scaled_dot = ttnn::where(mask.value()->get_value(), grad_scaled_dot, /* other */ 0.0F);
            }

            auto grad_q =
                ttnn::matmul(grad_scaled_dot, key->get_value(), /* transpose_a */ false, /* transpose_b */ false);
            grad_q = ttnn::multiply(grad_q, scale);

            auto grad_k =
                ttnn::matmul(grad_scaled_dot, query->get_value(), /* transpose_a */ true, /* transpose_b */ false);
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
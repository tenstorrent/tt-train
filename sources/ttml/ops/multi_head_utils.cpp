#include "multi_head_utils.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr heads_creation(autograd::TensorPtr x, uint32_t num_heads) {
    auto x_shape = x->get_value().get_shape();

    auto batch_size = x_shape[0];
    assert(x_shape[1] == 1U);
    auto sequence_length = x_shape[2];
    auto embedding_dim = x_shape[3];

    auto out = autograd::create_tensor();
    // initial - (B, 1, S, E)
    // (B, 1, E, S)
    auto result = ttnn::transpose(x->get_value(), -2, -1);
    // (B, H, E/H, S)
    result =
        ttnn::reshape(result, core::create_shape({batch_size, num_heads, embedding_dim / num_heads, sequence_length}));
    // (B, H, S, E/H)
    result = ttnn::transpose(result, -1, -2);
    out->set_value(result);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim]() {
        // (B, H, S, E/H)
        auto grad_output = out->get_grad();
        // (B, H, E/H, S)
        auto grad_result = ttnn::transpose(grad_output, -1, -2);
        // (B, 1, E, S)
        grad_result = ttnn::reshape(grad_result, core::create_shape({batch_size, 1, embedding_dim, sequence_length}));
        // (B, 1, S, E)
        grad_result = ttnn::transpose(grad_result, -2, -1);
        x->add_grad(grad_result);
    };

    auto links = autograd::get_links(x);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr heads_fusion(autograd::TensorPtr x) {
    auto x_shape = x->get_value().get_shape();

    uint32_t batch_size = x_shape[0];
    uint32_t num_heads = x_shape[1];
    assert(num_heads != 1U);
    uint32_t sequence_length = x_shape[2];
    uint32_t embedding_dim = x_shape[3];

    auto out = autograd::create_tensor();
    // (B, H, S, E/H)
    // (B, H, E/H, S)
    auto result = ttnn::transpose(x->get_value(), -2, -1);
    // (B, 1, E, S)
    result = ttnn::reshape(result, core::create_shape({batch_size, 1, embedding_dim * num_heads, sequence_length}));
    // (B, 1, S, E)
    result = ttnn::transpose(result, -2, -1);
    out->set_value(result);

    autograd::GradFunction grad = [out, x, num_heads, batch_size, sequence_length, embedding_dim]() {
        // (B, 1, S, E)
        auto grad_output = out->get_grad();
        // (B, 1, E, S)
        auto grad_result = ttnn::transpose(grad_output, -2, -1);
        // (B, H, E/H, S)
        grad_result =
            ttnn::reshape(grad_result, core::create_shape({batch_size, num_heads, embedding_dim, sequence_length}));
        // (B, H, S, E/H)
        grad_result = ttnn::transpose(grad_result, -1, -2);
        x->add_grad(grad_result);
    };

    auto links = autograd::get_links(x);
    out->set_node(ttml::autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
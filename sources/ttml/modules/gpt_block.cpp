#include "gpt_block.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

GPTMLP::GPTMLP(uint32_t embedding_size, float dropout_prob) {
    create_name("gpt_mlp");
    fc1 = std::make_shared<LinearLayer>(embedding_size, embedding_size * 4);
    fc2 = std::make_shared<LinearLayer>(embedding_size * 4, embedding_size);
    ln1 = std::make_shared<LayerNormLayer>(embedding_size * 4);
    dropout = std::make_shared<DropoutLayer>(dropout_prob);

    register_module(fc1, "fc1");
    register_module(fc2, "fc2");
    register_module(ln1, "ln1");
    register_module(dropout, "dropout");
}

autograd::TensorPtr GPTMLP::operator()(autograd::TensorPtr x) {
    x = (*fc1)(x);
    x = ops::gelu(x);
    x = (*ln1)(x);
    x = (*fc2)(x);
    x = (*dropout)(x);
    return x;
}

GPTBlock::GPTBlock(uint32_t embedding_size, uint32_t num_heads, float dropout_prob) {
    mlp = std::make_shared<GPTMLP>(embedding_size, dropout_prob);
    ln1 = std::make_shared<LayerNormLayer>(embedding_size);
    ln2 = std::make_shared<LayerNormLayer>(embedding_size);
    attention = std::make_shared<MultiHeadAttention>(embedding_size, num_heads, dropout_prob);

    create_name("gpt_block");
    register_module(mlp, "mlp");
    register_module(ln1, "ln1");
    register_module(ln2, "ln2");
    register_module(attention, "attention");
}

autograd::TensorPtr GPTBlock::operator()(autograd::TensorPtr x, const autograd::TensorPtr& mask) {
    auto residual = x;
    x = (*ln1)(x);
    // print_tensor_stats(x, "LN1");
    x = (*attention)(x, mask);
    // print_tensor_stats(x, "attention");
    x = ops::add(x, residual);
    // print_tensor_stats(x, "residual");

    residual = x;
    x = (*ln2)(x);
    // print_tensor_stats(x, "LN2");
    x = (*mlp)(x);
    // print_tensor_stats(x, "MLP");
    x = ops::add(x, residual);
    // print_tensor_stats(x, "residual");

    return x;
}

}  // namespace ttml::modules
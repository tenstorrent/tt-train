#include "multi_head_attention.hpp"

#include "ops/multi_head_utils.hpp"

namespace ttml::modules {

MultiHeadAttention::MultiHeadAttention(uint32_t embedding_dim_, uint32_t num_heads_, float dropout_prob) :
    embedding_dim(embedding_dim_), num_heads(num_heads_) {
    // create layers
    q_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    k_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    v_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);
    dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob);
    out_linear = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, embedding_dim);

    // register modules
    create_name("multi_head_attention");
    register_module(q_linear, "q_linear");
    register_module(k_linear, "k_linear");
    register_module(v_linear, "v_linear");
    register_module(dropout, "dropout");
    register_module(out_linear, "out_linear");
}

ttml::autograd::TensorPtr MultiHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto key = (*k_linear)(x);
    auto query = (*q_linear)(x);
    auto value = (*v_linear)(x);

    auto key_with_heads = ops::heads_creation(key, num_heads);
    auto query_with_heads = ops::heads_creation(query, num_heads);
    auto value_with_heads = ops::heads_creation(value, num_heads);

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*out_linear)(attention);
    out = (*dropout)(out);

    return out;
}

}  // namespace ttml::modules
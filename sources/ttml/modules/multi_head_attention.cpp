// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_head_attention.hpp"

#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

MultiHeadAttention::MultiHeadAttention(uint32_t embedding_dim_, uint32_t num_heads_, float dropout_prob_) :
    m_embedding_dim(embedding_dim_), m_num_heads(num_heads_) {
    // create layers
    m_query_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);
    m_key_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);
    m_value_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(dropout_prob_);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim);

    // register modules
    create_name("multi_head_attention");
    register_module(m_query_linear, "q_linear");
    register_module(m_key_linear, "k_linear");
    register_module(m_value_linear, "v_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
}

ttml::autograd::TensorPtr MultiHeadAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto key = (*m_key_linear)(x);
    auto query = (*m_query_linear)(x);
    auto value = (*m_value_linear)(x);

    auto key_with_heads = ops::heads_creation(key, m_num_heads);
    auto query_with_heads = ops::heads_creation(query, m_num_heads);
    auto value_with_heads = ops::heads_creation(value, m_num_heads);

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);

    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules

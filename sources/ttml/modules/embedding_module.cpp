#include "embedding_module.hpp"

#include <stdexcept>

#include "init/tensor_initializers.hpp"
#include "ops/embedding_op.hpp"

namespace ttml::modules {

void Embedding::initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim) {
    auto* device = &autograd::ctx().get_device();
    m_weight = autograd::create_tensor();
    init::normal_init(
        m_weight, core::create_shape({1, 1, num_embeddings, embedding_dim}), /* normal params */ {0.F, 1.F});
}

Embedding::Embedding(uint32_t num_embeddings, uint32_t embedding_dim) {
    if (num_embeddings % TILE_HEIGHT != 0) {
        throw std::logic_error("num_embeddings must be a multiple of TILE_HEIGHT");
    }
    if (embedding_dim % TILE_WIDTH == 0) {
        throw std::logic_error("embedding_dim must be a multiple of TILE_WIDTH");
    }
    initialize_tensors(num_embeddings, embedding_dim);

    create_name("embedding");
    register_tensor(m_weight, "weight");
}

autograd::TensorPtr Embedding::operator()(const autograd::TensorPtr& tensor) {
    auto sentence_size = tensor->get_value().get_shape()[-1];
    if (sentence_size % TILE_HEIGHT != 0 || sentence_size % TILE_WIDTH != 0) {
        throw std::logic_error("sentence_size must be a multiple of TILE_HEIGHT and TILE_WIDTH");
    }
    return ops::embedding_op(tensor, m_weight);
}

}  // namespace ttml::modules
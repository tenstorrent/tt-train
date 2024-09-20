#include "embedding_module.hpp"

#include "ops/embedding_op.hpp"

namespace ttml::modules {

void Embedding::initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim) {
    auto* device = &autograd::ctx().get_device();
    std::vector<float> weight_data((size_t)num_embeddings * embedding_dim);
    init::normal_init(weight_data, {0.F, 1.F});
    m_weight = autograd::create_tensor(
        core::from_vector(weight_data, core::create_shape({1, 1, num_embeddings, embedding_dim}), device));
}

Embedding::Embedding(uint32_t num_embeddings, uint32_t embedding_dim) {
    initialize_tensors(num_embeddings, embedding_dim);
}

autograd::TensorPtr Embedding::operator()(const autograd::TensorPtr& tensor) {
    return ops::embedding_op(tensor, m_weight);
}

}  // namespace ttml::modules
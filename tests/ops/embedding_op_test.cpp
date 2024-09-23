#include "ops/embedding_op.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/losses.hpp"

TEST(EmbeddingOpTest, EmbeddingForwardBackward) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = core::zeros(core::create_shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = core::from_vector<uint32_t>(
        input_data, core::create_shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    autograd::TensorPtr embeddings = ops::embedding_op(input, weight);

    std::vector<float> target_vector((size_t)batch_size * sentence_size * embedding_dim);
    for (uint32_t i = 0; i < batch_size * sentence_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            target_vector[embedding_dim * i + j] = static_cast<float>(i);
        }
    }
    auto target_tensor = autograd::create_tensor(
        core::from_vector(target_vector, core::create_shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto result = ttml::ops::mse_loss(target_tensor, embeddings);
    result->backward();

    auto weight_grad_tensor = weight->get_grad();
    auto weight_grad_data = core::to_vector(weight_grad_tensor);
    for (uint32_t i = 0; i < num_embeddings; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            EXPECT_NEAR(
                weight_grad_data[embedding_dim * i + j],
                -static_cast<float>(i) / sentence_size / embedding_dim / batch_size * 2.F,
                1e-2);
        }
    }
}

TEST(EmbeddingOpTest, EmbeddingBadShapes0_BROKEN) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 13;
    uint32_t embedding_dim = 26;
    auto weight_tensor = core::zeros(core::create_shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = core::from_vector<uint32_t>(
        input_data, core::create_shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    EXPECT_ANY_THROW(ops::embedding_op(input, weight));
}

TEST(EmbeddingOpTest, EmbeddingBadShapes1_BROKEN) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = core::zeros(core::create_shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 13;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = core::from_vector<uint32_t>(
        input_data, core::create_shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    EXPECT_ANY_THROW(ops::embedding_op(input, weight));
}

TEST(EmbeddingOpTest, EmbeddingBadLayout_BROKEN) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = core::zeros(core::create_shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor =
        core::from_vector<uint32_t>(input_data, core::create_shape({batch_size, 1, 1, sentence_size}), device);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    EXPECT_ANY_THROW(ops::embedding_op(input, weight));
}

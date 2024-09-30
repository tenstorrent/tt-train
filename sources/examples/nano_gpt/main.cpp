
#include <iostream>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_char_dataset.hpp"
#include "datasets/utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/linear_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, mask, positions
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryCharDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1) {
        m_sum += loss * static_cast<float>(count);
        m_count += count;
    }

    [[nodiscard]] float average() const {
        if (m_count == 0) {
            return 0.F;
        }
        return m_sum / static_cast<float>(m_count);
    }

    void reset() {
        m_sum = 0.0F;
        m_count = 0;
    }
};

std::string read_file_to_str(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

class BigramFCModel : public ttml::autograd::ModuleBase {
public:
    std::shared_ptr<ttml::modules::LinearLayer> fc1;
    std::shared_ptr<ttml::modules::Embedding> emb;

    BigramFCModel(uint32_t vocab_size, uint32_t num_tokens, uint32_t hidden_dim) {
        // make vocab_size divisible by 32
        vocab_size = (vocab_size + 31) / 32 * 32;

        // create layers
        emb = std::make_shared<ttml::modules::Embedding>(vocab_size, hidden_dim);
        fc1 = std::make_shared<ttml::modules::LinearLayer>(hidden_dim, num_tokens);

        create_name("bigram_fc_model");

        register_module(emb, "emb");
        register_module(fc1, "fc1");
    }

    ttml::autograd::TensorPtr operator()(
        ttml::autograd::TensorPtr x,
        [[maybe_unused]] ttml::autograd::TensorPtr positions,
        [[maybe_unused]] ttml::autograd::TensorPtr masks) const {
        x = (*emb)(x);
        x = (*fc1)(x);
        return x;
    }
};

class Transformer : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::Embedding> tok_emb;
    std::shared_ptr<ttml::modules::Embedding> pos_emb;
    std::vector<std::shared_ptr<ttml::modules::GPTBlock>> blocks;
    std::shared_ptr<ttml::modules::LinearLayer> fc;

public:
    Transformer(uint32_t vocab_size, uint32_t max_sequence_length) {
        uint32_t embedding_size = 128;
        uint32_t num_heads = 1;
        float dropout_prob = 0.F;
        uint32_t num_blocks = 1;

        uint32_t vocab_size_divisible = (vocab_size + 31) / 32 * 32;
        assert(vocab_size_divisible % 32 == 0);
        assert(max_sequence_length % 32 == 0);
        assert(embedding_size % 32 == 0);
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible, embedding_size);
        pos_emb = std::make_shared<ttml::modules::Embedding>(max_sequence_length, embedding_size);
        blocks.reserve(num_blocks);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            blocks.push_back(std::make_shared<ttml::modules::GPTBlock>(embedding_size, num_heads, dropout_prob));
        }
        fc = std::make_shared<ttml::modules::LinearLayer>(embedding_size, vocab_size);

        create_name("transformer");
        register_module(tok_emb, "tok_emb");
        register_module(pos_emb, "pos_emb");
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
        }
        register_module(fc, "fc");
    }

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& x,
        const ttml::autograd::TensorPtr& positions,
        const ttml::autograd::TensorPtr& mask) {
        auto tok_emb_out = (*tok_emb)(x);
        auto pos_emb_out = (*pos_emb)(positions);
        auto out = ttml::ops::add(tok_emb_out, pos_emb_out);
        for (auto& block : blocks) {
            out = (*block)(out, mask);
        }
        auto logits = (*fc)(out);
        return logits;
    }
};

int main() {
    const std::string data_folder = "/home/ubuntu/ML-Framework-CPP/sources/examples/nano_gpt/data";
    // const std::string data_path = data_folder + "/shakespeare.txt";
    const std::string data_path = data_folder + "/shakespeare_slice.txt";

    std::string text;
    try {
        text = read_file_to_str(data_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    uint32_t sequence_length = 32;
    auto [dataset, tokenizer] = ttml::datasets::create_in_memory_char_dataset(text, sequence_length);
    fmt::print("Dataset size: {}\n", dataset.get_size());
    fmt::print("Vocab size: {}\n", tokenizer.get_vocab_size());

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_async(true);
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, vocab_size = tokenizer.get_vocab_size(), device](std::vector<DatasetSample>&& samples) {
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> data;
            std::vector<float> targets;
            std::vector<uint32_t> positions;
            std::vector<float> mask;

            positions.reserve((size_t)batch_size * sequence_length);
            mask.reserve((size_t)batch_size * sequence_length * sequence_length);
            for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                for (int i = 0; i < sequence_length; ++i) {
                    positions.push_back(i);
                    for (int j = 0; j < sequence_length; ++j) {
                        mask.push_back(i >= j ? 1.0F : 0.0F);
                    }
                }
            }

            data.reserve((size_t)batch_size * sequence_length);
            targets.reserve((size_t)batch_size * vocab_size * sequence_length);
            std::vector<float> one_hot_target(vocab_size);
            for (auto& [features, target_span] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));

                for (auto target : target_span) {
                    std::fill(one_hot_target.begin(), one_hot_target.end(), 0.0F);
                    one_hot_target[target] = 1.0F;
                    std::copy(one_hot_target.begin(), one_hot_target.end(), std::back_inserter(targets));
                }
            }
            auto data_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t>(
                data, ttml::core::create_shape({batch_size, 1, 1, sequence_length}), device, Layout::ROW_MAJOR));
            auto targets_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
                targets, ttml::core::create_shape({batch_size, 1, sequence_length, vocab_size}), device));
            auto masks_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
                mask, ttml::core::create_shape({batch_size, 1, sequence_length, sequence_length}), device));
            auto positions_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t>(
                positions, ttml::core::create_shape({batch_size, 1, 1, sequence_length}), device, Layout::ROW_MAJOR));
            return std::make_tuple(data_tensor, targets_tensor, masks_tensor, positions_tensor);
        };

    uint32_t batch_size = 32;
    LossAverageMeter loss_meter;
    int global_step = 0;

    auto train_dataloader = DataLoader(dataset, /* batch_size */ batch_size, /* shuffle */ true, collate_fn);

    // auto model = BigramFCModel(tokenizer.get_vocab_size(), tokenizer.get_vocab_size(), /* hidden_dim */ 128);
    auto model = Transformer(tokenizer.get_vocab_size(), sequence_length);

    auto sgd_params = ttml::optimizers::SGDConfig();
    sgd_params.lr = 0.1;
    sgd_params.momentum = 0.9;
    // sgd_params.weight_decay = 0.0001;

    auto optimizer = ttml::optimizers::SGD(model.parameters(), sgd_params);
    const uint32_t num_epochs = 20;
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto [features, target, masks, positions] : train_dataloader) {
            optimizer.zero_grad();
            auto output = model(features, positions, masks);
            auto loss = ttml::ops::cross_entropy_loss(target, output);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            loss_meter.update(loss_float, features->get_value().get_shape()[0]);
            fmt::print("Step: {}, Loss: {}\n", global_step++, loss_float);
            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
        }
    }
    return 0;
}

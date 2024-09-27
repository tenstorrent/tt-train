
#include <iostream>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_char_dataset.hpp"
#include "datasets/utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryCharDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

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

    ttml::autograd::TensorPtr operator()(ttml::autograd::TensorPtr x) const {
        x = (*emb)(x);
        x = (*fc1)(x);
        return x;
    }
};

int main() {
    const std::string data_folder = "/home/ubuntu/ML-Framework-CPP/sources/examples/nano_gpt/data";
    const std::string data_path = data_folder + "/shakespeare.txt";

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

    auto* device = &ttml::autograd::ctx().get_device();
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, num_tokens = tokenizer.get_vocab_size(), device, &tokenizer](
            std::vector<DatasetSample>&& samples) {
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> data;
            std::vector<float> targets;
            data.reserve(static_cast<size_t>(batch_size * sequence_length));
            targets.reserve(static_cast<size_t>(batch_size * num_tokens * sequence_length));
            std::vector<float> one_hot_target(num_tokens);
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
                targets, ttml::core::create_shape({batch_size, 1, sequence_length, num_tokens}), device));
            return std::make_pair(data_tensor, targets_tensor);
        };

    uint32_t batch_size = 1;
    auto train_dataloader = DataLoader(dataset, /* batch_size */ batch_size, /* shuffle */ true, collate_fn);

    auto model = BigramFCModel(tokenizer.get_vocab_size(), tokenizer.get_vocab_size(), /* hidden_dim */ 128);

    auto sgd_params = ttml::optimizers::SGDConfig();
    sgd_params.lr = 0.1;
    sgd_params.momentum = 0.9;

    auto optimizer = ttml::optimizers::SGD(model.parameters(), sgd_params);
    for (auto [features, target] : train_dataloader) {
        optimizer.zero_grad();
        auto output = model(features);
        auto output_vec = ttml::core::to_vector(output->get_value());
        // fmt::print("Output: {}\n", output_vec);
        auto loss = ttml::ops::cross_entropy_loss(target, output);
        auto loss_float = ttml::core::to_vector(loss->get_value())[0];
        fmt::print("Loss: {}\n", loss_float);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }
    return 0;
}
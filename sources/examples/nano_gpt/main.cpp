#include <CLI/CLI.hpp>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_char_dataset.hpp"
#include "datasets/utils.hpp"
#include "models.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, mask, positions
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryCharDataset,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

struct DemoConfig {
    // training
    uint32_t batch_size = 64;
    uint32_t sequence_length = 256;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 5000;
    float dropout_prob = 0.2F;
    // model
    uint32_t num_heads = 6;
    uint32_t embedding_dim = 384;
    uint32_t num_blocks = 6;
    // optimizer
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
};
const DemoConfig config;

template <typename Model, typename Tokenizer>
void evaluate(
    const std::shared_ptr<Model> &model, const Tokenizer &tokenizer, uint32_t max_sequence_length, uint32_t num_heads) {
    model->eval();

    std::string prompt;
    fmt::print("Enter a prompt: ");
    std::getline(std::cin, prompt);

    if (prompt.empty()) {
        prompt = "\n";
    }

    auto *device = &ttml::autograd::ctx().get_device();
    auto prompt_tokens = tokenizer.encode(prompt);

    auto pad_token_id = 0U;

    auto vocab_size = tokenizer.get_vocab_size();
    auto tokens_to_generate = 1024U;
    for (uint32_t token_idx = 0; token_idx < tokens_to_generate; ++token_idx) {
        auto prompt_tokens_padded = prompt_tokens;
        while (prompt_tokens_padded.size() < max_sequence_length) {
            prompt_tokens_padded.push_back(pad_token_id);
        }

        if (prompt_tokens_padded.size() > max_sequence_length) {
            prompt_tokens_padded.erase(
                prompt_tokens_padded.begin(),
                prompt_tokens_padded.begin() + (prompt_tokens_padded.size() - max_sequence_length));
        }

        auto prompt_tokens_padded_size = static_cast<uint32_t>(prompt_tokens_padded.size());
        auto prompt_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t>(
            prompt_tokens_padded,
            ttml::core::create_shape({1, 1, 1, prompt_tokens_padded_size}),
            device,
            Layout::ROW_MAJOR));

        auto positions_vector = std::vector<uint32_t>(prompt_tokens_padded.size());
        std::iota(positions_vector.begin(), positions_vector.end(), 0);
        auto positions_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t>(
            positions_vector,
            ttml::core::create_shape({1, 1, 1, prompt_tokens_padded_size}),
            device,
            Layout::ROW_MAJOR));

        std::vector<float> mask;
        mask.reserve(static_cast<size_t>(prompt_tokens_padded_size * prompt_tokens_padded_size * num_heads));
        for (int head = 0; head < num_heads; ++head) {
            for (int i = 0; i < prompt_tokens_padded_size; ++i) {
                for (int j = 0; j < prompt_tokens_padded_size; ++j) {
                    mask.push_back(i >= j ? 1.0F : 0.0F);
                }
            }
        }

        auto mask_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
            mask,
            ttml::core::create_shape({1, num_heads, prompt_tokens_padded_size, prompt_tokens_padded_size}),
            device));

        auto output = (*model)(prompt_tensor, positions_tensor, mask_tensor);
        fmt::print("Output shape: {}\n", output->get_value().get_shape());
        auto output_vector = ttml::core::to_vector(output->get_value());

        uint32_t predicted_token_id = prompt_tokens_padded_size - 1U;
        auto logits_ptr = output_vector.data() + predicted_token_id * vocab_size;
        for (uint32_t idx = 0; idx < vocab_size; ++idx) {
            fmt::print("Token {} Logits[{}]: {}\n", tokenizer.decode({idx}), idx, logits_ptr[idx]);
        }
        auto token_ptr = std::max_element(logits_ptr, logits_ptr + vocab_size);
        auto token_id = static_cast<uint32_t>(std::distance(logits_ptr, token_ptr));
        prompt_tokens.push_back(token_id);

        auto str = tokenizer.decode(prompt_tokens);
        fmt::print("*******************\n");
        fmt::print("Generated text: {}\n", str);
        fmt::print("*******************\n");
    }

    model->train();
}

int main(int argc, char **argv) {
    CLI::App app{"NanoGPT Example"};
    argv = app.ensure_utf8(argv);

    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t max_steps = config.max_steps;
    uint32_t batch_size = config.batch_size;
    uint32_t sequence_length = config.sequence_length;
    std::string model_path = "/tmp/nano_gpt.msgpack";
    std::string data_path = "/home/ubuntu/ML-Framework-CPP/sources/examples/nano_gpt/data/shakespeare.txt";
    bool is_eval = false;

    app.add_option("-b,--batch_size", batch_size, "Batch size")->default_val(batch_size);
    app.add_option("-i,--model_save_interval", model_save_interval, "Model save interval")
        ->default_val(model_save_interval);
    app.add_option("-p,--model_path", model_path, "Model path")->default_val(model_path);
    app.add_option("-d,--data_path", data_path, "Data path")->default_val(data_path);
    app.add_option("-s,--seed", seed, "Seed")->default_val(seed);
    app.add_option("-m,--max_steps", max_steps, "Max steps")->default_val(max_steps);
    app.add_flag("-e,--eval", is_eval, "Evaluation mode")->default_val(is_eval);
    CLI11_PARSE(app, argc, argv);

    // set seed
    ttml::autograd::ctx().set_seed(seed);

    std::string text;
    try {
        text = read_file_to_str(data_path);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    fmt::print("Max steps {}\n", max_steps);
    fmt::print("Batch size {}\n", batch_size);
    fmt::print("Seed {}\n", ttml::autograd::ctx().get_seed());

    auto [dataset, tokenizer] = ttml::datasets::create_in_memory_char_dataset(text, sequence_length);
    fmt::print("Dataset size: {}\n", dataset.get_size());
    fmt::print("Vocab size: {}\n", tokenizer.get_vocab_size());

    auto *device = &ttml::autograd::ctx().get_device();

    // disable for now, unexpected freezes and crashes
    // device->enable_async(true);

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, num_heads = config.num_heads, vocab_size = tokenizer.get_vocab_size(), device](
            std::vector<DatasetSample> &&samples) {
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> data;
            std::vector<float> targets;
            std::vector<uint32_t> positions;
            std::vector<float> mask;

            positions.reserve((size_t)batch_size * sequence_length);
            for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                for (int i = 0; i < sequence_length; ++i) {
                    positions.push_back(i);
                }
            }

            mask.reserve((size_t)batch_size * sequence_length * sequence_length * num_heads);
            for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
                for (int head = 0; head < num_heads; ++head) {
                    for (int i = 0; i < sequence_length; ++i) {
                        for (int j = 0; j < sequence_length; ++j) {
                            mask.push_back(i >= j ? 1.0F : 0.0F);
                        }
                    }
                }
            }

            data.reserve((size_t)batch_size * sequence_length);
            targets.reserve((size_t)batch_size * vocab_size * sequence_length);
            std::vector<float> one_hot_target(vocab_size);
            for (auto &[features, target_span] : samples) {
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
                mask, ttml::core::create_shape({batch_size, num_heads, sequence_length, sequence_length}), device));
            auto positions_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t>(
                positions, ttml::core::create_shape({batch_size, 1, 1, sequence_length}), device, Layout::ROW_MAJOR));
            return std::make_tuple(data_tensor, targets_tensor, masks_tensor, positions_tensor);
        };

    LossAverageMeter loss_meter;
    auto train_dataloader = DataLoader(dataset, /* batch_size */ batch_size, /* shuffle */ true, collate_fn);

    auto transformer_config = TransformerConfig();
    transformer_config.num_heads = config.num_heads;
    transformer_config.embedding_dim = config.embedding_dim;
    transformer_config.dropout_prob = config.dropout_prob;
    transformer_config.num_blocks = config.num_blocks;
    transformer_config.vocab_size = tokenizer.get_vocab_size();
    transformer_config.max_sequence_length = sequence_length;
    auto model = std::make_shared<Transformer>(transformer_config);

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;
    fmt::print("AdamW configuration:\n");
    fmt::print("    Learning rate: {}\n", adamw_params.lr);
    fmt::print("    Weight decay: {}\n", adamw_params.weight_decay);
    auto optimizer = ttml::optimizers::AdamW(model->parameters(), adamw_params);

    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        fmt::print("Loading model from {}\n", model_path);
        load_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
        fmt::print("Model loaded after {} steps\n", optimizer.get_steps());
    }

    if (is_eval) {
        fmt::print("Evaluation started\n");
        evaluate(model, tokenizer, sequence_length, config.num_heads);
        fmt::print("Evaluation finished\n");
        return 0;
    }

    const uint32_t num_epochs = config.num_epochs;
    std::ofstream loss_file("loss.txt");
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto [features, target, masks, positions] : train_dataloader) {
            optimizer.zero_grad();
            auto output = (*model)(features, positions, masks);
            auto loss = ttml::ops::cross_entropy_loss(target, output);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            loss_meter.update(loss_float, features->get_value().get_shape()[0]);
            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();

            auto global_step = optimizer.get_steps();
            fmt::print("Step: {}, Loss: {}\n", global_step, loss_float);
            loss_file << fmt::format("Step: {}, Loss: {}", global_step, loss_float) << std::endl;

            if (!model_path.empty() && global_step % model_save_interval == 0) {
                save_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
            }

            if (global_step >= max_steps) {
                break;
            }
        }
        if (optimizer.get_steps() >= max_steps) {
            break;
        }
    }

    if (!model_path.empty()) {
        save_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
    }

    loss_file.close();
    return 0;
}

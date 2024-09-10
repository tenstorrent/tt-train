
#include <iostream>
#include <ttnn/tensor/tensor.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"

using ttml::autograd::TensorPtr;

int main() {
    const size_t training_samples_count = 100000;
    [[maybe_unused]] const size_t test_samples_count = 1000;
    const size_t num_features = 8;
    const size_t num_targets = 2;
    const float noise = 0.F;
    const bool bias = true;

    auto training_params = ttml::datasets::MakeRegressionParams{
        .n_samples = training_samples_count,
        .n_features = num_features,
        .n_targets = num_targets,
        .noise = noise,
        .bias = bias,
    };

    auto training_dataset = ttml::datasets::make_regression(training_params);

    auto* device = &ttml::autograd::ctx().get_device();

    using DatasetSample = std::pair<std::vector<float>, std::vector<float>>;
    std::function<std::pair<tt::tt_metal::Tensor, tt::tt_metal::Tensor>(const std::vector<DatasetSample>& samples)>
        collate_fn = [&num_features, &num_targets, device](const std::vector<DatasetSample>& samples) {
            const uint32_t batch_size = samples.size();
            std::vector<float> data;
            std::vector<float> targets;
            data.reserve(batch_size * num_features);
            targets.reserve(batch_size * num_targets);
            for (const auto& [features, target] : samples) {
                std::move(features.begin(), features.end(), std::back_inserter(data));
                std::move(target.begin(), target.end(), std::back_inserter(targets));
            }

            auto data_tensor = ttml::core::from_vector(
                data, ttnn::Shape(std::array<uint32_t, 4>{batch_size, 1, 1, num_features}), device);
            auto targets_tensor = ttml::core::from_vector(
                targets, ttnn::Shape(std::array<uint32_t, 4>{batch_size, 1, 1, num_targets}), device);
            return std::make_pair(data_tensor, targets_tensor);
        };

    auto train_dataloader = ttml::datasets::DataLoader<
        ttml::datasets::InMemoryFloatVecDataset,
        decltype(collate_fn),
        std::pair<tt::tt_metal::Tensor, tt::tt_metal::Tensor>>(training_dataset, 128, true, collate_fn);

    auto model = ttml::modules::LinearLayer(num_features, num_targets);

    auto sgd_config = ttml::optimizers::SGDConfig{.lr = 1.0F, .momentum = 0.F};
    auto optimizer = ttml::optimizers::SGD(model.parameters(), sgd_config);

    int training_step = 0;
    for (auto [data, targets] : train_dataloader) {
        auto data_ptr = std::make_shared<ttml::autograd::Tensor>(data);
        auto targets_ptr = std::make_shared<ttml::autograd::Tensor>(targets);

        optimizer.zero_grad();
        auto output = model(data_ptr);
        auto loss = ttml::ops::mse_loss(targets_ptr, output);
        auto loss_float = ttml::core::to_vector(loss->get_value())[0];
        std::cout << "Step: " << (training_step++) << " Loss: " << loss_float << '\n';
        loss->backward();
        optimizer.step();
    }
}
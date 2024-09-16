
#include <iostream>
#include <mnist/mnist_reader.hpp>
#include <ttnn/operations/eltwise/ternary/where.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_dataset.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::vector<uint8_t>, uint8_t>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t>,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

template <typename Model>
float evaluate(DataLoader& test_dataloader, Model& model, size_t num_targets) {
    float num_correct = 0;
    float num_samples = 0;
    for (const auto& [data, target] : test_dataloader) {
        auto output = model(data);
        auto output_vec = ttml::core::to_vector(output->get_value());
        auto target_vec = ttml::core::to_vector(target->get_value());
        for (size_t i = 0; i < output_vec.size(); i += num_targets) {
            auto predicted_class = std::distance(
                output_vec.begin() + i,
                std::max_element(output_vec.begin() + i, output_vec.begin() + (i + num_targets)));
            auto target_class = std::distance(
                target_vec.begin() + i,
                std::max_element(target_vec.begin() + i, target_vec.begin() + (i + num_targets)));
            num_correct += static_cast<float>(predicted_class == target_class);
            num_samples++;
        }
    }
    return num_correct / num_samples;
};

template <typename Model>
void clip_gradient_norm_(Model& model, float max_norm) {
    for (auto& [name, param] : model.parameters()) {
        auto& grad = param->get_grad();
        auto grad_squared = ttnn::multiply(grad, grad);

        ttnn::Shape shape(std::array<uint32_t, 4>{1, 1, 1, 1});
        auto out = ttml::core::from_vector({0.F}, shape, &ttml::autograd::ctx().get_device());
        ttnn::moreh_sum(grad_squared, std::nullopt, true, out, grad_squared.memory_config());
        auto grad_norm_tensor = ttnn::sqrt(out);

        auto grad_norm_tensor_mask = ttnn::ge(grad_norm_tensor, max_norm);
        auto inv_scaled_grad_norm_tensor = ttnn::divide(grad_norm_tensor, max_norm);

        auto grad_legacy_shape = grad.get_legacy_shape();
        auto inv_scaled_grad_norm_tensor_legacy_shape = inv_scaled_grad_norm_tensor.get_legacy_shape();
        const size_t rank = grad.get_shape().rank();
        assert(rank == 4);
        std::vector<uint32_t> repeat_shape_array(rank);
        for (size_t i = 0; i < rank; ++i) {
            repeat_shape_array[i] = grad_legacy_shape[i] / inv_scaled_grad_norm_tensor_legacy_shape[i];
        }
        tt::tt_metal::Shape repeat_legacy_shape = tt::tt_metal::Shape(repeat_shape_array, grad_legacy_shape.padding());
        auto repeat_shape = ttnn::Shape(repeat_legacy_shape);

        inv_scaled_grad_norm_tensor = ttnn::repeat(inv_scaled_grad_norm_tensor, repeat_shape);
        auto scaled_grads = ttnn::divide(grad, inv_scaled_grad_norm_tensor);
        fmt::print("Shape of scaled_grads: {}\n", scaled_grads.get_shape());

        auto grad_norm_tensor_mask_reshaped = ttnn::repeat(grad_norm_tensor_mask, repeat_shape);
        auto clipped_grad = ttnn::where(grad_norm_tensor_mask_reshaped, scaled_grads, grad);

        param->set_grad(clipped_grad);
    }
};

int main() {
    // Load MNIST data
    const size_t num_targets = 10;
    const size_t num_features = 784;
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t> training_dataset(
        dataset.training_images, dataset.training_labels);
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t> test_dataset(
        dataset.test_images, dataset.test_labels);

    auto* device = &ttml::autograd::ctx().get_device();
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [num_features, num_targets, device](std::vector<DatasetSample>&& samples) {
            const uint32_t batch_size = samples.size();
            std::vector<float> data;
            std::vector<float> targets;
            data.reserve(batch_size * num_features);
            targets.reserve(batch_size * num_targets);
            for (auto& [features, target] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));

                std::vector<float> one_hot_target(num_targets, 0.0F);
                one_hot_target[target] = 1.0F;
                std::copy(one_hot_target.begin(), one_hot_target.end(), std::back_inserter(targets));
            }

            std::transform(data.begin(), data.end(), data.begin(), [](float pixel) { return pixel / 255.0F - 0.5F; });

            auto data_tensor = std::make_shared<ttml::autograd::Tensor>(ttml::core::from_vector(
                data, ttnn::Shape(std::array<uint32_t, 4>{batch_size, 1, 1, num_features}), device));
            auto targets_tensor = std::make_shared<ttml::autograd::Tensor>(ttml::core::from_vector(
                targets, ttnn::Shape(std::array<uint32_t, 4>{batch_size, 1, 1, num_targets}), device));
            return std::make_pair(data_tensor, targets_tensor);
        };

    const uint32_t batch_size = 128;
    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);
    auto test_dataloader = DataLoader(test_dataset, batch_size, /* shuffle */ false, collate_fn);

    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .m_input_features = num_features, .m_hidden_features = {128}, .m_output_features = num_targets};
    auto model = ttml::modules::MultiLayerPerceptron(model_params);

    // evaluate model before training (sanity check to get reasonable accuracy 1/num_targets)
    // float accuracy_before_training = evaluate(test_dataloader, model, num_targets);
    // fmt::print("Accuracy before training: {}\n", evaluate(test_dataloader, model, num_targets));

    const float learning_rate = 0.1F * (batch_size / 128.F);
    const float momentum = 0.9F;
    const float weight_decay = 0.F;
    auto sgd_config =
        ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = momentum, .weight_decay = weight_decay};

    fmt::print("SGD configuration:\n");
    fmt::print("    Learning rate: {}\n", sgd_config.lr);
    fmt::print("    Momentum: {}\n", sgd_config.momentum);
    fmt::print("    Dampening {}\n", sgd_config.dampening);
    fmt::print("    Weight decay: {}\n", sgd_config.weight_decay);
    fmt::print("    Nesterov: {}\n", sgd_config.nesterov);
    auto optimizer = ttml::optimizers::SGD(model.parameters(), sgd_config);

    LossAverageMeter loss_meter;
    const int logging_interval = 50;

    int training_step = 0;
    const size_t num_epochs = 10;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto& [data, target] : train_dataloader) {
            optimizer.zero_grad();
            auto output = model(data);
            auto loss = ttml::ops::cross_entropy_loss(target, output);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            loss_meter.update(loss_float, batch_size);
            if (training_step % logging_interval == 0) {
                fmt::print("Step: {:5d} | Average Loss: {:.4f}\n", training_step, loss_meter.average());
            }
            loss->backward();
            clip_gradient_norm_(model, 1.0F);
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
            training_step++;
        }
        const float test_accuracy = evaluate(test_dataloader, model, num_targets);
        fmt::print(
            "Epoch: {:3d} | Average Loss: {:.4f} | Accuracy: {:.4f}\n", epoch + 1, loss_meter.average(), test_accuracy);
        loss_meter.reset();
    }
}
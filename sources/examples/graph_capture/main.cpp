
#include <iostream>

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
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

using ttml::autograd::TensorPtr;

int main() {
    const size_t num_targets = 10;
    const uint32_t batch_size = 128;
    const size_t num_features = 784;
    auto* device = &ttml::autograd::ctx().get_device();

    auto batch = std::make_shared<ttml::autograd::Tensor>(
        ttml::core::zeros(ttnn::Shape{std::array<uint32_t, 4>{batch_size, 1, 1, num_features}}, device));
    auto target = std::make_shared<ttml::autograd::Tensor>(
        ttml::core::zeros(ttnn::Shape{std::array<uint32_t, 4>{batch_size, 1, 1, num_targets}}, device));

    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .m_input_features = num_features, .m_hidden_features = {128}, .m_output_features = num_targets};
    auto model = ttml::modules::MultiLayerPerceptron(model_params);

    auto mode = tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH;
    ttnn::graph::GraphProcessor graph_processor(mode);
    graph_processor.begin_graph_capture(mode);
    auto output = model(batch);
    auto loss = ttml::ops::cross_entropy_loss(target, output);
    auto forward_trace = graph_processor.end_graph_capture();

    auto call = [&] {
        loss->backward();
        return 0;
    };
    auto backward_trace = ttnn::graph::query_trace(call);

    auto pretty_forward_trace = forward_trace.dump(4);
    auto pretty_backward_trace = backward_trace.dump(4);

    const std::string path = "/home/ubuntu/graph_traces/";
    std::ofstream forward_trace_file(fmt::format("{}/forward_trace.json", path));
    forward_trace_file << pretty_forward_trace;
    forward_trace_file.close();

    std::ofstream backward_trace_file(fmt::format("{}/backward_trace.json", path));
    backward_trace_file << pretty_backward_trace;
    backward_trace_file.close();

    fmt::print("Forward trace saved to: {}/forward_trace.json\n", path);
    fmt::print("Backward trace saved to: {}/backward_trace.json\n", path);
    fmt::print("Capture complete\n");

    return 0;
}
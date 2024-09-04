#include "binary_ops.hpp"

#include <ttnn/tensor/types.hpp>
#include <vector>

#include "autograd/graph.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {
ttml::autograd::TensorPtr operator+(const autograd::TensorPtr& a, const autograd::TensorPtr& b) {
    autograd::TensorPtr out;

    out->set_value(ttnn::add(a->get_value(), b->get_value()));
    autograd::GradFunction grad = [a, b, out]() {
        tt::tt_metal::MemoryConfig mem_config;

        // uint8_t queue_id,
        // const Tensor& grad,
        // const Tensor& input,
        // const Tensor& other,
        // const MemoryConfig& output_mem_config,
        // const std::vector<bool>& are_required_outputs,
        // std::optional<Tensor> input_grad,
        // std::optional<Tensor> other_grad
        auto res = ttnn::add_bw(out->get_grad(), a->get_value(), b->get_value(), mem_config);

        a->add_grad(res[0]);
        b->add_grad(res[1]);
    };
    std::vector<autograd::NodeId> links;

    const auto& a_node = a->get_node();
    const auto& b_node = b->get_node();
    if (a_node.has_value()) {
        links.push_back(a_node.value());
    }

    if (b_node.has_value()) {
        links.push_back(b_node.value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}
}  // namespace ttml::ops

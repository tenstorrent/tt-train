#include "ops/unary_ops.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out;
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_grad(), mem_config);

        tensor->add_grad(res[0]);
    };

    std::vector<autograd::NodeId> links;
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
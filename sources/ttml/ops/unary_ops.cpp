#include "ops/unary_ops.hpp"

#include <optional>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
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

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::gelu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        static const std::string approx_mode = "tanh";
        auto res = ttnn::gelu_bw(out->get_grad(), tensor->get_grad(), approx_mode, mem_config);

        tensor->add_grad(res[0]);
    };

    std::vector<autograd::NodeId> links;
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::mean(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        const auto inv_volume = 1.0F / static_cast<float>(tensor->get_value().get_shape().volume());
        // TODO: remove multiply in favor of ttnn::repeat
        auto res = ttnn::multiply(ttnn::ones_like(tensor->get_value()), ttnn::multiply(out->get_grad(), inv_volume));
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links;
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr sum(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::sum(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        // TODO: remove multiply in favor of ttnn::repeat
        auto res = ttnn::multiply(ttnn::ones_like(tensor->get_value()), out->get_grad());
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links;
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim) {
    if (new_batch_dim == 1 || tensor->get_value().shape()[0] == new_batch_dim) {
        return tensor;
    }
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    ttnn::Shape repeats(std::array<uint32_t, 4>{new_batch_dim, 1, 1, 1});
    // currently assuming tensor came with shape: {1,X,Y,Z} and we want to get {B,X,Y,Z}
    out->set_value(ttnn::repeat(tensor->get_value(), repeats));

    autograd::GradFunction grad = [tensor, out]() {
        // TODO: remove multiply in favor of ttnn::repeat
        auto res = ttnn::sum(out->get_value(), 0);
        auto output_shape = tensor->get_value().get_legacy_shape();
        res = res.reshape(output_shape);
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links;
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
#include "ops/unary_ops.hpp"

#include <array>
#include <optional>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_value(), mem_config);
        tensor->add_grad(res[0]);
    };

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    autograd::TensorPtr out = std::make_shared<autograd::Tensor>();
    out->set_value(ttnn::gelu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        static const std::string approx_mode = "tanh";
        auto res = ttnn::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode, mem_config);
        assert(res.size() == 1U && "Gelu backward should return only one gradient");
        tensor->add_grad(res.front().value());
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    ttnn::Shape shape(std::array<uint32_t, 4>{1, 1, 1, 1});
    autograd::TensorPtr out =
        std::make_shared<autograd::Tensor>(core::from_vector({0.F}, shape, &autograd::ctx().get_device()));
    ttnn::moreh_mean(
        tensor->get_value(), std::nullopt, true, std::nullopt, out->get_value(), std::nullopt, std::nullopt);
    autograd::GradFunction grad = [tensor, out]() {
        auto resulting_shape = tensor->get_value().get_shape();
        auto res = ttnn::moreh_mean_backward(
            out->get_grad(), std::nullopt, false, resulting_shape, std::nullopt, std::nullopt, std::nullopt);
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links = autograd::get_links(tensor);

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
        auto res = ttnn_fixed::sum_over_batch(out->get_grad());
        tensor->add_grad(res);
    };
    std::vector<autograd::NodeId> links = autograd::get_links(tensor);

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
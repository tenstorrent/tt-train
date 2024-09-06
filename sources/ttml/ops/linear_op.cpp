#include "linear_op.hpp"

#include "autograd/auto_context.hpp"

namespace ttml::ops {

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    autograd::TensorPtr out;
    out->set_value(ttnn::linear(
        tensor->get_value(), weight->get_value(), bias->get_value(), /* transpose_a */ false, /* tranpose_b */ true));

    autograd::GradFunction grad = [weight, bias, tensor, out]() {
        /// TODO: implement backward
    };

    std::vector<autograd::NodeId> links;
    if (weight->get_node().has_value()) {
        links.push_back(weight->get_node().value());
    }
    if (bias->get_node().has_value()) {
        links.push_back(bias->get_node().value());
    }
    if (tensor->get_node().has_value()) {
        links.push_back(tensor->get_node().value());
    }

    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
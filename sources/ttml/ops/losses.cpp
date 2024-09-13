#include "losses.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr mse_loss(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction, ReduceType reduce) {
    auto diff = ops::sub(target, prediction);  // TODO: @rfurko-tt use "ttnn::squared_difference"
    auto diff_2 = ops::mul(diff, diff);  // TODO: need to add backward "ttnn::squared_difference_bw" might be faster
    if (reduce == ReduceType::MEAN) {
        return ops::mean(diff_2);
    } else {
        throw std::logic_error("Unsupported MSE reduction type");
    }
}

autograd::TensorPtr cross_entropy_loss_without_reduce_(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction) {
    const float eps = 1e-6F;
    auto prediction_tensor = ttnn::softmax(prediction->get_value(), -1);
    prediction_tensor = ttnn::clip(prediction_tensor, eps, 1.0F - eps);
    auto loss = ttnn::multiply(target->get_value(), ttnn::log(prediction_tensor));
    loss = ttnn::neg(loss);
    loss = ttnn::multiply(loss, loss.get_shape()[-1]);

    auto out = std::make_shared<autograd::Tensor>();
    out->set_value(loss);
    autograd::GradFunction grad = [target, prediction_tensor, prediction, out]() {
        auto grad = ttnn::subtract(prediction_tensor, target->get_value());
        grad = ttnn::multiply(grad, out->get_grad());
        prediction->add_grad(grad);
    };

    auto links = autograd::get_links(prediction);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

autograd::TensorPtr cross_entropy_loss(
    const autograd::TensorPtr& target, const autograd::TensorPtr& prediction, ReduceType reduce) {
    auto loss = cross_entropy_loss_without_reduce_(target, prediction);
    if (reduce == ReduceType::MEAN) {
        return ops::mean(loss);
    } else {
        throw std::logic_error("Unsupported cross entropy reduction type");
    }
}

}  // namespace ttml::ops
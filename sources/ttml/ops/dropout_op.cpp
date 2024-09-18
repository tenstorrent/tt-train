#include "dropout_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ttml::ops {
autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability) {
    auto mask = core::ones_like(tensor->get_value());
    mask = ttnn::dropout(mask, autograd::ctx().get_generator()(), probability, 1.0F / (1.0F - probability));
    auto out = autograd::create_tensor();
    auto masked_out = ttnn::multiply(tensor->get_value(), mask);
    out->set_value(masked_out);
    autograd::GradFunction grad = [tensor, out, mask]() {
        auto res = ttnn::multiply(out->get_grad(), mask);
        tensor->add_grad(res);
    };

    std::vector<autograd::NodeId> links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}
}  // namespace ttml::ttml::ops
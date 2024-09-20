#include "embedding_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

autograd::TensorPtr embedding_op(const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight) {
    auto embeddings =
        ttnn::embedding(tensor->get_value(), weight->get_value(), /* pad_token */ std::nullopt, Layout::TILE);
    auto out = autograd::create_tensor(embeddings);

    autograd::GradFunction grad = [tensor, weight, out]() {
        // there is not gradient flowing back to the input tensor, only to weights
        auto weight_grad = ttnn::embedding_bw(tensor->get_value(), weight->get_value(), out->get_grad());
        weight->add_grad(weight_grad);
    };

    auto links = autograd::get_links(weight);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
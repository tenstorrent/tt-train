#include "core/ttnn_all_includes.hpp"

namespace ttml::autograd {

void clip_tensor_norm_(tt::tt_metal::Tensor& tensor, const float max_norm);

template <typename Model>
void clip_gradient_norm_(Model& model, float max_norm) {
    for (auto& [name, param] : model.parameters()) {
        auto& grad = param->get_grad();
        clip_tensor_norm_(grad, max_norm);
    }
};

}  // namespace ttml::autograd
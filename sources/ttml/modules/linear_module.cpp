#include "linear_module.hpp"

#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "init/cpu_initializers.hpp"
#include "init/ttnn_tensor_initializers.hpp"

namespace ttml::modules {

void LinearLayer::initialize_tensors(uint32_t in_features, uint32_t out_features) {
    auto* device = &autograd::ctx().get_device();
    tt::tt_metal::Shape weight_shape({1, 1, out_features, in_features});
    std::vector<float> weight_vec((size_t)out_features * in_features);
    init::xavier_uniform_init(weight_vec, init::FanParams{in_features, out_features});
    m_weight = std::make_shared<autograd::Tensor>(core::from_vector(weight_vec, ttnn::Shape(weight_shape), device));

    tt::tt_metal::Shape bias_shape({1, 1, 1, out_features});
    auto bias = core::zeros(ttnn::Shape(bias_shape), device);
    m_bias = std::make_shared<autograd::Tensor>(bias);
}

LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features) {
    initialize_tensors(in_features, out_features);

    create_name("linear");
    register_tensor(m_weight, "weight");
    register_tensor(m_bias, "bias");
}

autograd::TensorPtr LinearLayer::operator()(const autograd::TensorPtr& tensor) {
    return ops::linear_op(tensor, m_weight, m_bias);
}

}  // namespace ttml::modules
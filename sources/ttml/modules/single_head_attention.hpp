#include "autograd/tensor.hpp"
#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "ops/scaled_dot_product_attention.hpp"

namespace ttml::modules {

class SingleHeadAttention : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::LinearLayer> q_linear;
    std::shared_ptr<ttml::modules::LinearLayer> k_linear;
    std::shared_ptr<ttml::modules::LinearLayer> v_linear;
    std::shared_ptr<ttml::modules::LinearLayer> out_linear;
    std::shared_ptr<ttml::modules::DropoutLayer> dropout;

public:
    explicit SingleHeadAttention(uint32_t embedding_dim, float dropout_prob = 0.2);

    ttml::autograd::TensorPtr operator()(const ttml::autograd::TensorPtr& x);
};

}  // namespace ttml::modules
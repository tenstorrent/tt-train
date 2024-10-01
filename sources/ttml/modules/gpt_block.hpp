#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "modules/single_head_attention.hpp"

namespace ttml::modules {

class GPTMLP : public autograd::ModuleBase {
    std::shared_ptr<LinearLayer> fc1;
    std::shared_ptr<LinearLayer> fc2;
    std::shared_ptr<LayerNormLayer> ln1;
    std::shared_ptr<DropoutLayer> dropout;

public:
    explicit GPTMLP(uint32_t embedding_size, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& x);
};

class GPTBlock : public autograd::ModuleBase {
    std::shared_ptr<GPTMLP> mlp;
    std::shared_ptr<LayerNormLayer> ln1;
    std::shared_ptr<LayerNormLayer> ln2;
    std::shared_ptr<MultiHeadAttention> attention;

public:
    explicit GPTBlock(uint32_t embedding_size, uint32_t num_heads, float dropout_prob);

    autograd::TensorPtr operator()(const autograd::TensorPtr& input, const autograd::TensorPtr& mask);
};

}  // namespace ttml::modules
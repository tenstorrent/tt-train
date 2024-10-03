
#include "autograd/module_base.hpp"
#include "core/ttnn_all_includes.hpp"

namespace ttml::optimizers {

struct AdamWConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{0.01F};
    // TODO: add amsgrad
};

class AdamW {
public:
    AdamW(autograd::NamedParameters parameters, const AdamWConfig& config);

    void zero_grad();

    void step();

private:
    size_t steps{0};
    AdamWConfig m_config;
    ttml::autograd::NamedParameters m_parameters;
    std::unordered_map<std::string, tt::tt_metal::Tensor> m_first_moment;
    std::unordered_map<std::string, tt::tt_metal::Tensor> m_second_moment;
};

}  // namespace ttml::optimizers
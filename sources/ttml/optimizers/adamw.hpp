
#include "autograd/module_base.hpp"
#include "core/ttnn_all_includes.hpp"
#include "optimizer_base.hpp"

namespace ttml::optimizers {

struct AdamWConfig {
    float lr{1e-3F};
    float beta1{0.9F};
    float beta2{0.999F};
    float epsilon{1e-8F};
    float weight_decay{0.01F};
    // TODO: add amsgrad
};

class AdamW : public OptimizerBase {
public:
    AdamW(autograd::NamedParameters parameters, const AdamWConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] autograd::NamedParameters get_state_dict() const override;
    void set_state_dict(const autograd::NamedParameters& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

private:
    size_t m_steps{0};
    AdamWConfig m_config;
    ttml::autograd::NamedParameters m_parameters;
    autograd::NamedParameters m_first_moment;
    autograd::NamedParameters m_second_moment;
};

}  // namespace ttml::optimizers
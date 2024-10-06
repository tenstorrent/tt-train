#pragma once

#include <ttnn/tensor/tensor.hpp>

#include "autograd/module_base.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::optimizers {

class IOptimizer {
public:
    IOptimizer() = default;
    IOptimizer(const IOptimizer&) = delete;
    IOptimizer& operator=(const IOptimizer&) = delete;
    IOptimizer(IOptimizer&&) = delete;
    IOptimizer& operator=(IOptimizer&&) = delete;
    virtual ~IOptimizer() = default;

    virtual void zero_grad() = 0;

    virtual void step() = 0;

    [[nodiscard]] virtual autograd::NamedParameters get_state_dict() const = 0;
    virtual void set_state_dict(autograd::NamedParameters dict) = 0;

    [[nodiscard]] virtual size_t get_steps() const = 0;
    virtual void set_steps(size_t steps) = 0;
};

}  // namespace ttml::optimizers
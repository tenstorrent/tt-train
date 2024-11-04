#pragma once

#include <optional>

#include "core/ttnn_all_includes.hpp"

namespace ttml::autograd {

class AutocastTensor {
    bool m_is_float32 = false;
    tt::tt_metal::Tensor m_half_precision_tensor;
    tt::tt_metal::Tensor m_full_precision_tensor;

public:
    AutocastTensor() = default;
    explicit AutocastTensor(const tt::tt_metal::Tensor &tensor);
    AutocastTensor(const AutocastTensor &) = default;
    AutocastTensor(AutocastTensor &&) noexcept = default;
    AutocastTensor &operator=(const AutocastTensor &) = default;
    AutocastTensor &operator=(AutocastTensor &&) noexcept = default;
    ~AutocastTensor() = default;

    [[nodiscard]] ttnn::Shape get_shape() const;
    void set_tensor(const tt::tt_metal::Tensor &tensor);
    const tt::tt_metal::Tensor &get_tensor(bool half_precision = true) const;
    tt::tt_metal::Tensor &get_tensor(bool half_precision = true);
};

}  // namespace ttml::autograd

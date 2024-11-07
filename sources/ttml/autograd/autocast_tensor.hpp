#pragma once

#include <cstdint>
#include <optional>

#include "core/ttnn_all_includes.hpp"

namespace ttml::autograd {

enum class Precision : uint8_t { HALF = 0, FULL = 1 };

class AutocastTensor {
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
    const tt::tt_metal::Tensor &get_tensor(Precision precision = Precision::HALF) const;

    // mutable tensor always returns full precision tensor
    // otherwise we wouldn't be able to guarantee consistency between full and half precision tensors
    tt::tt_metal::Tensor &get_mutable_tensor();
};

}  // namespace ttml::autograd

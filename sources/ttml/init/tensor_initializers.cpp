#include "tensor_initializers.hpp"

#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "core/tensor_utils.hpp"
#include "cpu_initializers.hpp"

namespace ttml::init {
void uniform_init(tt::tt_metal::Tensor& t, UniformRange range) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    uniform_init(vec, range);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void normal_init(tt::tt_metal::Tensor& t, NormalParams params) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    normal_init(vec, params);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void constant_init(tt::tt_metal::Tensor& t, float value) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    constant_init(vec, value);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void xavier_uniform_init(tt::tt_metal::Tensor& t, FanParams params) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    xavier_uniform_init(vec, params);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void xavier_normal_init(tt::tt_metal::Tensor& t, FanParams params) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    xavier_normal_init(vec, params);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void kaiming_uniform_init(tt::tt_metal::Tensor& t, int fan_in) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    kaiming_uniform_init(vec, fan_in);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}

void kaiming_normal_init(tt::tt_metal::Tensor& t, int fan_out) {
    auto shape = t.get_legacy_shape();
    auto* device = t.device();
    assert(device);
    size_t volume = tt::tt_metal::compute_volume(shape);
    std::vector<float> vec(volume);
    kaiming_normal_init(vec, fan_out);

    auto new_tensor = ttml::core::from_vector(vec, shape, device);

    ttnn::assign(t, new_tensor);
}
}  // namespace ttml::init
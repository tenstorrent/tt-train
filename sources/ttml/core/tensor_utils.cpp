#include "core/tensor_utils.hpp"

#include <fmt/color.h>

#include <common/bfloat16.hpp>
#include <optional>
#include <stdexcept>
#include <ttnn/operations/creation.hpp>

#include "core/ttnn_all_includes.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) { return ttnn::zeros_like(tensor); }

void fill(tt::tt_metal::Tensor& tensor, const float value) {
    // TODO: optimize to do it in one operation
    tensor = ttnn::multiply(ttnn::ones_like(tensor), value);
}

tt::tt_metal::Tensor zeros(const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device) {
    return ttnn::zeros(ttnn::Shape(shape), std::nullopt, std::nullopt, *device);
}
tt::tt_metal::Tensor ones(const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device) {
    return ttnn::ones(ttnn::Shape(shape), std::nullopt, std::nullopt, *device);
}

tt::tt_metal::Tensor from_vector(
    const std::vector<float>& buffer, const tt::tt_metal::Shape& shape, tt::tt_metal::Device* device) {
    const Layout layout = Layout::ROW_MAJOR;
    const DataType data_type = DataType::FLOAT32;
    MemoryConfig output_mem_config{};
    size_t volume = tt::tt_metal::compute_volume(shape);
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }
    auto owned_buffer = tt::tt_metal::owned_buffer::create<float>(buffer.size());

    for (auto idx = 0; auto value : buffer) {
        owned_buffer[idx++] = value;
        idx++;
    }
    auto output = tt::tt_metal::Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
    if (device != nullptr) {
        output = output.to(layout);
        output = output.to(device, output_mem_config);
    }
    return output;
}

std::vector<float> to_vector(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffer = tt::tt_metal::host_buffer::get_as<float>(cpu_tensor);
    std::vector<float> out(buffer.size());

    for (size_t idx = 0; auto& it : buffer) {
        out[idx] = it;
        idx++;
    }

    return out;
}

}  // namespace ttml::core
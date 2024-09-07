#include "core/tensor_utils.hpp"

#include <fmt/color.h>

#include <common/bfloat16.hpp>
#include <optional>
#include <stdexcept>
#include <ttnn/operations/creation.hpp>

#include "core/ttnn_all_includes.hpp"
#include "ttnn/cpp/ttnn/operations/core/core.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

namespace {

// copypaste from deprecated tensor pybinds ttnn
tt::tt_metal::OwnedBuffer create_owned_buffer_from_vector_of_floats(

    const std::vector<float>& data, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::BFLOAT4_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp4_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return tt::tt_metal::owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::FLOAT32: {
            auto data_copy = data;
            return tt::tt_metal::owned_buffer::create<float>(std::move(data_copy));
        }
        case DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(std::begin(data), std::end(data), std::begin(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return tt::tt_metal::owned_buffer::create<bfloat16>(std::move(bfloat16_data));
        }
        default: {
            throw std::runtime_error("Cannot create a host buffer!");
        }
    }
}
}  // namespace
namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) { return ttnn::zeros_like(tensor); }

tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor) { return ttnn::ones_like(tensor); }

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
    const Layout layout = Layout::TILE;
    const DataType data_type = DataType::BFLOAT16;
    MemoryConfig output_mem_config{};
    size_t volume = tt::tt_metal::compute_volume(shape);
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }
    auto owned_buffer = create_owned_buffer_from_vector_of_floats(buffer, data_type);
    auto output = tt::tt_metal::Tensor(OwnedStorage{owned_buffer}, shape, data_type, Layout::ROW_MAJOR);
    if (device != nullptr) {
        output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        output = ttnn::to_device(output, device, output_mem_config);
    }
    return output;
}

std::vector<float> to_vector(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    auto buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(cpu_tensor);
    auto shape = tensor.get_shape();
    size_t real_size = shape.volume();
    std::vector<float> out(real_size);

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = buffer[i].to_float();
    }

    return out;
}

}  // namespace ttml::core
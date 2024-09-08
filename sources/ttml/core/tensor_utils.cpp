#include "core/tensor_utils.hpp"

#include <fmt/color.h>

#include <common/bfloat16.hpp>
#include <cstddef>
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

std::vector<float> untile_tensor(
    const std::vector<float>& tiled_data, const ttnn::Shape& untiled_shape, const ttnn::Shape& tiled_shape) {
    std::vector<float> untiled_data(
        static_cast<size_t>(untiled_shape[0] * untiled_shape[1] * untiled_shape[2] * untiled_shape[3]));

    for (int a = 0; a < untiled_shape[0]; ++a) {
        for (int b = 0; b < untiled_shape[1]; ++b) {
            for (int c = 0; c < untiled_shape[2]; ++c) {
                for (int d = 0; d < untiled_shape[3]; ++d) {
                    // Compute the index for the untiled tensor
                    int untiled_index = ((a * untiled_shape[1] + b) * untiled_shape[2] + c) * untiled_shape[3] + d;

                    // Compute the index in the tiled data
                    int tiled_index = ((a * tiled_shape[1] + b) * tiled_shape[2] + c) * tiled_shape[3] + d;

                    // Extract the data from the tiled buffer
                    untiled_data[untiled_index] = tiled_data[tiled_index];
                }
            }
        }
    }

    return untiled_data;
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
    // cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);

    auto buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(cpu_tensor);
    auto shape = tensor.get_shape();
    std::vector<float> out(buffer.size());

    for (size_t i = 0; i < out.size(); i++) {
        out[i] = buffer[i].to_float();
    }
    auto final_res = untile_tensor(out, shape, shape.with_tile_padding());
    return final_res;
}

}  // namespace ttml::core
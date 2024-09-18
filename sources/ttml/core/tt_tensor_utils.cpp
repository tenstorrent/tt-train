#include "tt_tensor_utils.hpp"

#include <fmt/color.h>

#include <common/bfloat16.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <ttnn/operations/creation.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>

#include "ttnn_all_includes.hpp"

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

// TODO: add support for other types of data
// TODO: optimize precomputing multipliers
std::vector<float> untile_tensor_to_vec(const tt::tt_metal::Tensor& cpu_tensor) {
    auto tiled_buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(cpu_tensor);
    auto untiled_shape = cpu_tensor.get_shape();
    auto tiled_shape = untiled_shape.with_tile_padding();

    // Calculate total size of the untiled tensor
    size_t total_size = 1;
    for (uint32_t i = 0; i < untiled_shape.rank(); ++i) {
        total_size *= untiled_shape[i];
    }

    std::vector<float> untiled_data(total_size);

    auto compute_flat_index = [](const std::vector<uint32_t>& indices, ttnn::Shape& shape) -> uint32_t {
        uint32_t flat_index = 0;
        uint32_t multiplier = 1;
        for (int i = (int)indices.size() - 1; i >= 0; --i) {
            flat_index += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return flat_index;
    };

    std::vector<uint32_t> indices(tiled_shape.rank(), 0);

    for (size_t idx = 0; idx < total_size; ++idx) {
        uint32_t untiled_index = compute_flat_index(indices, untiled_shape);
        uint32_t tiled_index = compute_flat_index(indices, tiled_shape);

        untiled_data[untiled_index] = tiled_buffer[tiled_index].to_float();

        for (int dim = (int)tiled_shape.rank() - 1; dim >= 0; --dim) {
            if (++indices[dim] < untiled_shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return untiled_data;
}

}  // namespace
namespace ttml::core {

tt::tt_metal::Tensor zeros_like(const tt::tt_metal::Tensor& tensor) { return ttnn::zeros_like(tensor); }

tt::tt_metal::Tensor ones_like(const tt::tt_metal::Tensor& tensor) { return ttnn::ones_like(tensor); }

tt::tt_metal::Tensor full(const ttnn::Shape& shape, float value, tt::tt_metal::Device* device) {
    auto padded = shape.with_tile_padding();
    // if the shape is not divisible by TILE_SIZE, we need to add padding
    if (padded[2] % ttnn::types::TILE_SIZE != 0 || padded[3] % ttnn::types::TILE_SIZE != 0) {
        int additional_padding_h =
            (ttnn::types::TILE_SIZE - (int)padded[2] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
        int additional_padding_w =
            (ttnn::types::TILE_SIZE - (int)padded[3] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
        auto padded_shape = ttnn::Shape(
            {shape[0], shape[1], shape[2], shape[3]},
            {
                padded[0],
                padded[1],
                (padded[2] + additional_padding_h),
                (padded[3] + additional_padding_w),
            });
        return ttnn::full(padded_shape, value, DataType::BFLOAT16, Layout::TILE, std::ref(*device));
    }
    // if not padding available, we can just create a tensor with the given shape
    return ttnn::full(shape, value, DataType::BFLOAT16, Layout::TILE, std::ref(*device));
}

tt::tt_metal::Tensor zeros(const ttnn::Shape& shape, tt::tt_metal::Device* device) {
    return core::full(shape, 0.F, device);
}

tt::tt_metal::Tensor ones(const ttnn::Shape& shape, tt::tt_metal::Device* device) {
    return core::full(shape, 1.F, device);
}

tt::tt_metal::Tensor from_vector(
    const std::vector<float>& buffer, const ttnn::Shape& shape, tt::tt_metal::Device* device) {
    const Layout layout = Layout::TILE;
    const DataType data_type = DataType::BFLOAT16;
    MemoryConfig output_mem_config{};
    size_t volume = tt::tt_metal::compute_volume(shape);
    if (buffer.size() != volume) {
        throw std::logic_error(
            fmt::format("Current buffer size is {} different from shape volume {}", buffer.size(), volume));
    }
    auto owned_buffer = create_owned_buffer_from_vector_of_floats(buffer, data_type);
    // remove possible paddings from the shape (it conflicts with ROW MAJOR)
    auto unpadded_shape = core::create_shape(shape);
    auto output = tt::tt_metal::Tensor(OwnedStorage{owned_buffer}, unpadded_shape, data_type, Layout::ROW_MAJOR);
    if (device != nullptr) {
        output = ttnn::to_layout(output, layout, std::nullopt, output_mem_config, device);
        output = ttnn::to_device(output, device, output_mem_config);
    }
    return output;
}

std::vector<float> to_vector(const tt::tt_metal::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu();
    cpu_tensor = cpu_tensor.to(Layout::ROW_MAJOR);

    auto buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(cpu_tensor);
    auto final_res = untile_tensor_to_vec(cpu_tensor);
    return final_res;
}

bool is_tensor_initialized(const tt::tt_metal::Tensor& tensor) { return tensor.tensor_attributes != nullptr; }

ttnn::Shape create_shape(const ttnn::Shape& shape) {
    auto legacy_shape = tt::tt_metal::LegacyShape({shape[0], shape[1], shape[2], shape[3]});
    return ttnn::Shape(legacy_shape);
}

ttnn::Shape create_shape(const std::array<uint32_t, 4>& args) { return ttnn::Shape{args}; }
}  // namespace ttml::core
#include "serialization.hpp"

#include <cstdint>

#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "msgpack_file.hpp"

namespace ttml::serialization {

// trivial type to the std::array of bytes
template <typename T>
std::string to_bytes(const T& value) {
    std::string bytes(sizeof(T), '\0');
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor) {
    auto shape = tensor.get_shape();
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto storage_type = tensor.storage_type();

    file.put(std::string(name) + "/shape", to_bytes(shape));
    file.put(std::string(name) + "/data_type", static_cast<int>(data_type));
    file.put(std::string(name) + "/layout", static_cast<int>(layout));
    file.put(std::string(name) + "/layout", static_cast<int>(storage_type));

    if (data_type == tt::tt_metal::DataType::BFLOAT16) {
        auto data = ttml::core::to_vector<float>(tensor);
        file.put(std::string(name) + "/data", std::span<const float>(data.data(), data.size()));
    } else if (data_type == tt::tt_metal::DataType::UINT32) {
        auto data = ttml::core::to_vector<uint32_t>(tensor);
        file.put(std::string(name) + "/data", std::span<const uint32_t>(data.data(), data.size()));
    }
}

void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor) {
    ttnn::Shape shape;
    tt::tt_metal::DataType data_type;
}

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::Tensor& tensor, bool save_grads) {}

void read_autograd_tensor(MsgPackFile& file, std::string_view name, const ttml::autograd::Tensor& tensor) {}

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params) {}
void read_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params) {}

}  // namespace ttml::serialization
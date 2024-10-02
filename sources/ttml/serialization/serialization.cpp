#include "serialization.hpp"

#include <cstdint>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "msgpack_file.hpp"

namespace ttml::serialization {

// trivial type to the std::string
template <typename T>
std::string to_bytes(const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    std::string bytes(sizeof(T), '\0');
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

template <typename T>
void from_bytes(const std::string& bytes, T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");

    if (bytes.size() != sizeof(T)) {
        throw std::invalid_argument("Invalid byte size for conversion to type T.");
    }
    std::memcpy(&value, bytes.data(), sizeof(T));
}

template <typename T>
void get_enum(MsgPackFile& file, std::string_view name, T& value) {
    int int_value = 0;
    file.get(std::string(name), int_value);
    value = static_cast<T>(int_value);
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
    tt::tt_metal::DataType data_type{};
    tt::tt_metal::Layout layout{};
    tt::tt_metal::StorageType storage_type{};

    auto shape = tensor.get_shape();
    std::string bytes;
    file.get(std::string(name) + "/shape", bytes);
    from_bytes<ttnn::Shape>(bytes, shape);

    get_enum(file, std::string(name) + "/data_type", data_type);
    get_enum(file, std::string(name) + "/layout", layout);
    get_enum(file, std::string(name) + "/layout", storage_type);

    if (data_type == tt::tt_metal::DataType::BFLOAT16) {
        std::vector<float> data;
        file.get(std::string(name) + "/data", data);
        tensor = core::from_vector(data, shape, &ttml::autograd::ctx().get_device(), layout);
    } else if (data_type == tt::tt_metal::DataType::UINT32) {
        std::vector<uint32_t> data;
        file.get(std::string(name) + "/data", data);
        tensor = core::from_vector(data, shape, &ttml::autograd::ctx().get_device(), layout);
    }
}

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::TensorPtr& tensor, bool save_grads) {
    write_ttnn_tensor(file, std::string(name) + "/value", tensor->get_value());
    file.put(std::string(name) + "/save_grads", save_grads);

    if (save_grads) {
        write_ttnn_tensor(file, std::string(name) + "/grad", tensor->get_grad());
    }
}

void read_autograd_tensor(MsgPackFile& file, std::string_view name, ttml::autograd::TensorPtr& tensor) {
    tt::tt_metal::Tensor value;
    bool save_grads = false;

    read_ttnn_tensor(file, std::string(name) + "/value", value);
    tensor->set_value(value);
    file.get(std::string(name) + "/save_grads", save_grads);

    if (save_grads) {
        tt::tt_metal::Tensor grad;
        read_ttnn_tensor(file, std::string(name) + "/grad", grad);
        tensor->set_grad(grad);
    }
}

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params) {
    for (const auto& [key, value] : params) {
        write_autograd_tensor(file, std::string(name) + "/" + key, value);
    }
}
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::autograd::NamedParameters& params) {
    for (const auto& [key, value] : params) {
        ttml::autograd::TensorPtr tensor;
        read_autograd_tensor(file, std::string(name) + "/" + key, tensor);
    }
}

}  // namespace ttml::serialization
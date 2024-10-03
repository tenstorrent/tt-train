#pragma once

#include <string_view>

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/ttnn_fwd.hpp"

namespace ttml::optimizers {
class SGD;
}
namespace ttml::serialization {
class MsgPackFile;

void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor);
void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor);

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::TensorPtr& tensor, bool save_grads = false);
void read_autograd_tensor(MsgPackFile& file, std::string_view name, ttml::autograd::TensorPtr& tensor);

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params);
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::autograd::NamedParameters& params);

void write_sgd_optimizer(MsgPackFile& file, std::string_view name, const ttml::optimizers::SGD& optimizer);
void read_sgd_optimizer(MsgPackFile& file, std::string_view name, ttml::optimizers::SGD& optimizer);

template <class T>
requires std::is_base_of_v<autograd::ModuleBase, T> void write_module(
    MsgPackFile& file, std::string_view name, const std::shared_ptr<T>& module) {
    write_module(file, name, module.get());
}

template <class T>
requires std::is_base_of_v<autograd::ModuleBase, T> void write_module(
    MsgPackFile& file, std::string_view name, const T* module) {
    auto named_parameters = module->parameters();
    write_named_parameters(file, name, named_parameters);
}

template <class T>
requires std::is_base_of_v<autograd::ModuleBase, T> void read_module(
    MsgPackFile& file, std::string_view name, const std::shared_ptr<T>& module) {
    read_module(file, name, module.get());
}

template <class T>
requires std::is_base_of_v<autograd::ModuleBase, T> void read_module(
    MsgPackFile& file, std::string_view name, T* module) {
    auto named_parameters = module->parameters();
    read_named_parameters(file, name, named_parameters);
}

}  // namespace ttml::serialization
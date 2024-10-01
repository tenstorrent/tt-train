#pragma once

#include <string_view>

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "core/ttnn_fwd.hpp"

namespace ttml::serialization {
class MsgPackFile;
void write_ttnn_tensor(MsgPackFile& file, std::string_view name, const tt::tt_metal::Tensor& tensor);
void read_ttnn_tensor(MsgPackFile& file, std::string_view name, tt::tt_metal::Tensor& tensor);

void write_autograd_tensor(
    MsgPackFile& file, std::string_view name, const ttml::autograd::Tensor& tensor, bool save_grads = false);
void read_autograd_tensor(MsgPackFile& file, std::string_view name, ttml::autograd::Tensor& tensor);

void write_named_parameters(MsgPackFile& file, std::string_view name, const ttml::autograd::NamedParameters& params);
void read_named_parameters(MsgPackFile& file, std::string_view name, ttml::autograd::NamedParameters& params);

}  // namespace ttml::serialization
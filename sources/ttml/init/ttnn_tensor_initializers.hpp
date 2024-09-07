#pragma once
#include "core/ttnn_fwd.hpp"
#include "init/cpu_initializers.hpp"
namespace ttml::init {
void uniform_init(tt::tt_metal::Tensor& t, UniformRange range);

void normal_init(tt::tt_metal::Tensor& t, NormalParams params);

void constant_init(tt::tt_metal::Tensor& t, float value);

void xavier_uniform_init(tt::tt_metal::Tensor& t, FanParams params);

void xavier_normal_init(tt::tt_metal::Tensor& t, FanParams params);

void kaiming_uniform_init(tt::tt_metal::Tensor& t, int fan_in);

void kaiming_normal_init(tt::tt_metal::Tensor& t, int fan_out);

}  // namespace ttml::init
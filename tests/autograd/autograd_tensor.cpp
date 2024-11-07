// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

using namespace ttml;

TEST(AutogradTensorTest, AutogradTensorFLOAT32) {
    auto tensor = autograd::create_tensor(
        core::ones(core::create_shape({1, 1, 1, 32}), &autograd::ctx().get_device(), DataType::FLOAT32));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), DataType::FLOAT32);
}

TEST(AutogradTensorTest, AutogradTensorBFLOAT16) {
    auto tensor = autograd::create_tensor(
        core::ones(core::create_shape({1, 1, 1, 32}), &autograd::ctx().get_device(), DataType::BFLOAT16));
    const auto& half_precision_tensor = tensor->get_value();
    const auto& full_precision_tensor = tensor->get_value(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), DataType::BFLOAT16);
}

TEST(AutogradTensorTest, AutocastTensor) {
    auto tt_tensor = core::ones(core::create_shape({1, 1, 1, 32}), &autograd::ctx().get_device(), DataType::FLOAT32);
    auto autocast_tensor = autograd::AutocastTensor(tt_tensor);
    const auto& half_precision_tensor = autocast_tensor.get_tensor();
    const auto& full_precision_tensor = autocast_tensor.get_tensor(autograd::PreferredPrecision::FULL);

    EXPECT_EQ(half_precision_tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(full_precision_tensor.dtype(), DataType::FLOAT32);
}

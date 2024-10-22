// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundefined-inline"
#pragma GCC diagnostic ignored "-Wdeprecated-volatile"
#pragma GCC diagnostic ignored "-Wdeprecated-this-capture"

#include <common/bfloat16.hpp>                                                                // NOLINT
#include <tests/tt_metal/test_utils/env_vars.hpp>                                             // NOLINT
#include <tt_metal/common/base_types.hpp>                                                     // NOLINT
#include <tt_metal/common/math.hpp>                                                           // NOLINT
#include <tt_metal/host_api.hpp>                                                              // NOLINT
#include <tt_metal/hostdevcommon/common_values.hpp>                                           // NOLINT
#include <tt_metal/impl/device/device.hpp>                                                    // NOLINT
#include <ttnn/core.hpp>                                                                      // NOLINT
#include <ttnn/cpp/ttnn/operations/core/core.hpp>                                             // NOLINT
#include <ttnn/device.hpp>                                                                    // NOLINT
#include <ttnn/operations/core/to_dtype/to_dtype_op.hpp>                                      // NOLINT
#include <ttnn/operations/creation.hpp>                                                       // NOLINT
#include <ttnn/operations/data_movement/pad/pad.hpp>                                          // NOLINT
#include <ttnn/operations/data_movement/permute/permute.hpp>                                  // NOLINT
#include <ttnn/operations/data_movement/repeat/repeat.hpp>                                    // NOLINT
#include <ttnn/operations/data_movement/slice/slice.hpp>                                      // NOLINT
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>  // NOLINT
#include <ttnn/operations/data_movement/transpose/transpose.hpp>                              // NOLINT
#include <ttnn/operations/data_movement/untilize/untilize.hpp>                                // NOLINT
#include <ttnn/operations/eltwise/binary/binary.hpp>                                          // NOLINT
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>                        // NOLINT
#include <ttnn/operations/eltwise/unary/unary.hpp>                                            // NOLINT
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>                                  // NOLINT
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>                          // NOLINT
#include <ttnn/operations/embedding/embedding.hpp>                                            // NOLINT
#include <ttnn/operations/embedding_backward/embedding_backward.hpp>                          // NOLINT
#include <ttnn/operations/matmul/matmul.hpp>                                                  // NOLINT
#include <ttnn/operations/moreh/moreh_adamw/moreh_adamw.hpp>                                  // NOLINT
#include <ttnn/operations/moreh/moreh_layer_norm/moreh_layer_norm.hpp>                        // NOLINT
#include <ttnn/operations/moreh/moreh_layer_norm_backward/moreh_layer_norm_backward.hpp>      // NOLINT
#include <ttnn/operations/moreh/moreh_linear_backward/moreh_linear_backward.hpp>              // NOLINT
#include <ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp>                                // NOLINT
#include <ttnn/operations/moreh/moreh_mean/moreh_mean.hpp>                                    // NOLINT
#include <ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp>                  // NOLINT
#include <ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss.hpp>                            // NOLINT
#include <ttnn/operations/moreh/moreh_nll_loss_backward/moreh_nll_loss_backward.hpp>          // NOLINT
#include <ttnn/operations/moreh/moreh_sum/moreh_sum.hpp>                                      // NOLINT
#include <ttnn/operations/normalization/softmax/softmax.hpp>                                  // NOLINT
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>                           // NOLINT
#include <ttnn/tensor/host_buffer/functions.hpp>                                              // NOLINT
#include <ttnn/tensor/tensor.hpp>                                                             // NOLINT
#include <ttnn/tensor/types.hpp>                                                              // NOLINT
#include <ttnn/types.hpp>                                                                     // NOLINT

#pragma GCC diagnostic pop

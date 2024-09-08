#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundefined-inline"
#pragma GCC diagnostic ignored "-Wdeprecated-volatile"
#pragma GCC diagnostic ignored "-Wdeprecated-this-capture"

#include <common/bfloat16.hpp>                                                                // NOLINT
#include <tests/tt_metal/test_utils/env_vars.hpp>                                             // NOLINT
#include <tt_metal/common/math.hpp>                                                           // NOLINT
#include <tt_metal/host_api.hpp>                                                              // NOLINT
#include <tt_metal/hostdevcommon/common_values.hpp>                                           // NOLINT
#include <tt_metal/impl/device/mesh_device.hpp>                                               // NOLINT
#include <ttnn/core.hpp>                                                                      // NOLINT
#include <ttnn/cpp/ttnn/operations/core/core.hpp>                                             // NOLINT
#include <ttnn/device.hpp>                                                                    // NOLINT
#include <ttnn/operations/creation.hpp>                                                       // NOLINT
#include <ttnn/operations/data_movement/repeat/repeat.hpp>                                    // NOLINT
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>  // NOLINT
#include <ttnn/operations/eltwise/binary/binary.hpp>                                          // NOLINT
#include <ttnn/operations/eltwise/binary_backward/binary_backward.hpp>                        // NOLINT
#include <ttnn/operations/eltwise/unary/unary.hpp>                                            // NOLINT
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>                          // NOLINT
#include <ttnn/operations/matmul/matmul.hpp>                                                  // NOLINT
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>                           // NOLINT
#include <ttnn/tensor/host_buffer/functions.hpp>                                              // NOLINT
#include <ttnn/types.hpp>                                                                     // NOLINT
#pragma GCC diagnostic pop
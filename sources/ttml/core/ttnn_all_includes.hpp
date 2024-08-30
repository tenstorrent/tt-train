#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundefined-inline"
#pragma GCC diagnostic ignored "-Wdeprecated-volatile"
#pragma GCC diagnostic ignored "-Wdeprecated-this-capture"

#include "common/bfloat16.hpp"                                                                // NOLINT
#include "tests/tt_metal/test_utils/env_vars.hpp"                                             // NOLINT
#include "tt_metal/host_api.hpp"                                                              // NOLINT
#include "tt_metal/hostdevcommon/common_values.hpp"                                           // NOLINT
#include "tt_metal/impl/device/device_mesh.hpp"                                               // NOLINT
#include "ttnn/core.hpp"                                                                      // NOLINT
#include "ttnn/device.hpp"                                                                    // NOLINT
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"  // NOLINT
#include "ttnn/operations/eltwise/binary/binary.hpp"                                          // NOLINT
#include "ttnn/operations/eltwise/unary/unary.hpp"                                            // NOLINT
#include "ttnn/types.hpp"                                                                     // NOLINT

#pragma GCC diagnostic pop
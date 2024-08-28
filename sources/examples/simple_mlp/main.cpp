
#include <iostream>

#include "common/bfloat16.hpp"                       // NOLINT
#include "tests/tt_metal/test_utils/env_vars.hpp"    // NOLINT
#include "tt_metal/host_api.hpp"                     // NOLINT
#include "tt_metal/hostdevcommon/common_values.hpp"  // NOLINT
#include "tt_metal/impl/device/device_mesh.hpp"      // NOLINT
#include "ttml.hpp"
#include "ttnn/core.hpp"                                                                      // NOLINT
#include "ttnn/device.hpp"                                                                    // NOLINT
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"  // NOLINT
#include "ttnn/operations/eltwise/binary/binary.hpp"                                          // NOLINT
#include "ttnn/operations/eltwise/unary/unary.hpp"                                            // NOLINT
#include "ttnn/types.hpp"                                                                     // NOLINT

int main() {
    std::srand(0);
    auto arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    std::cout << "Arch:" << tt::test_utils::get_env_arch_name() << std::endl;
    std::cout << "num_devices:" << num_devices_ << std::endl;
    auto device = tt::tt_metal::CreateDevice(0);
    std::cout << "Device created" << std::endl;
    tt::tt_metal::CloseDevice(device);
}
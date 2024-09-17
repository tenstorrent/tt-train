#pragma once

namespace tt::tt_metal {
struct Tensor;
class CommandQueue;
struct MemoryConfig;
class Device;
class DeviceMesh;
class LegacyShape;
}  // namespace tt::tt_metal

namespace ttnn {
using Tensor = tt::tt_metal::Tensor;  // not sure if it works but we can use original tensor namespace

}  // namespace ttnn
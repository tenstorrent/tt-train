#pragma once

#include <memory>

#include "ttnn_fwd.hpp"

namespace ttml::core {
// should I implement pimpl or its fine
class Device {
   public:
    explicit Device(int device_index);
    Device(Device&& device) = default;
    Device(const Device&) = delete;

    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = default;
    ~Device();

    [[nodiscard]] tt::tt_metal::Device& get_device();

   private:
    std::unique_ptr<tt::tt_metal::Device, void (*)(tt::tt_metal::Device*)> m_device;
};
}  // namespace ttml::core
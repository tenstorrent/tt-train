#pragma once

#include <memory>
#include <unordered_map>

#include "tensor.hpp"

namespace ttml::autograd {
class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;
class ModuleBase : public std::enable_shared_from_this<ModuleBase> {
   private:
    std::unordered_map<std::string, TensorPtr> m_named_tensors;
    std::unordered_map<std::string, ModuleBasePtr> m_named_modules;
};

}  // namespace ttml::autograd
#pragma once

#include <memory>
#include <unordered_map>

#include "tensor.hpp"

namespace ttml::autograd {
class ModuleBase : public std::enable_shared_from_this<ModuleBase> {
   private:
    std::unordered_map<std::string, Tensor> m_named_tensors;
    std::unordered_map<std::string, std::shared_ptr<ModuleBase>> m_named_modules;
};

}  // namespace ttml::autograd
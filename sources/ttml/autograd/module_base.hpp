#pragma once

#include <memory>
#include <unordered_map>

#include "tensor.hpp"

namespace ttml::autograd {
class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;
using NamedParameters = std::unordered_map<std::string, TensorPtr>;

class ModuleBase : public std::enable_shared_from_this<ModuleBase> {
private:
    std::string m_name;
    std::unordered_map<std::string, TensorPtr> m_named_tensors;
    std::unordered_map<std::string, ModuleBasePtr> m_named_modules;

protected:
    void create_name(const std::string& prefix);
    void register_tensor(const TensorPtr& tensor_ptr, const std::string& name);
    void register_module(const ModuleBasePtr& module_ptr, const std::string& name);

public:
    [[nodiscard]] const std::string& get_name() const;
    [[nodiscard]] NamedParameters parameters() const;
};

}  // namespace ttml::autograd
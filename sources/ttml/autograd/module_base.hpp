#pragma once

#include <memory>
#include <unordered_map>

#include "tensor.hpp"

namespace ttml::autograd {
class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;

class ModuleBase : public std::enable_shared_from_this<ModuleBase> {
private:
    std::string m_name;
    std::unordered_map<std::string, TensorPtr> m_named_tensors;
    std::unordered_map<std::string, ModuleBasePtr> m_named_modules;
    static size_t generate_id();

protected:
    void create_name(const std::string& prefix);
    void register_tensor(const std::string& name, TensorPtr tensor_ptr);
    void register_module(const std::string& name, ModuleBasePtr module_ptr);

public:
    const std::string& get_name() const;
    std::unordered_map<std::string, TensorPtr> parameters();
};

}  // namespace ttml::autograd
#include "module_base.hpp"

#include "auto_context.hpp"

namespace ttml::autograd {

void ModuleBase::register_tensor(const TensorPtr& tensor_ptr, const std::string& name) {
    auto [_, is_inserted] = m_named_tensors.emplace(name, tensor_ptr);
    if (!is_inserted) {
        throw std::logic_error("Names of two tensors coincide");
    }
}

void ModuleBase::register_module(const ModuleBasePtr& module_ptr, const std::string& name) {
    auto [_, is_inserted] = m_named_modules.emplace(name, module_ptr);
    if (!is_inserted) {
        throw std::logic_error("Names of two modules coincide");
    }
}

void ModuleBase::create_name(const std::string& prefix) {
    m_name = prefix + std::to_string(autograd::ctx().generate_module_id());
}

const std::string& ModuleBase::get_name() const { return m_name; }

NamedParameters ModuleBase::parameters() const {
    NamedParameters params;

    std::queue<std::pair<const ModuleBase*, std::string>> modules_to_process;
    modules_to_process.emplace(this, get_name() + "/");

    std::unordered_set<std::string> modules_in_queue;
    modules_in_queue.insert(get_name());
    while (!modules_to_process.empty()) {
        auto [module_ptr, name_prefix] = modules_to_process.front();
        modules_to_process.pop();

        for (const auto& [tensor_name, tensor_ptr] : module_ptr->m_named_tensors) {
            params.emplace(name_prefix + tensor_name, tensor_ptr);
        }

        for (const auto& [module_name, next_module_ptr] : m_named_modules) {
            if (!modules_in_queue.contains(next_module_ptr->get_name())) {
                modules_to_process.emplace(next_module_ptr.get(), name_prefix + module_name + "/");
                modules_in_queue.insert(next_module_ptr->get_name());
            }
        }
    }

    return params;
}

}  // namespace ttml::autograd
#pragma once

#include <memory>
#include <unordered_map>

#include "tensor.hpp"

namespace ttml::autograd {

enum class TrainMode { TRAIN, EVAL };

class ModuleBase;
using ModuleBasePtr = std::shared_ptr<ModuleBase>;
using NamedParameters = std::unordered_map<std::string, TensorPtr>;

class ModuleBase : public std::enable_shared_from_this<ModuleBase> {
private:
    std::string m_name;
    TrainMode m_train_mode = TrainMode::EVAL;

    std::unordered_map<std::string, TensorPtr> m_named_tensors;
    std::unordered_map<std::string, ModuleBasePtr> m_named_modules;

protected:
    void create_name(const std::string& prefix);
    void register_tensor(const TensorPtr& tensor_ptr, const std::string& name);
    void register_module(const ModuleBasePtr& module_ptr, const std::string& name);

public:
    ModuleBase() = default;
    virtual ~ModuleBase() = default;
    ModuleBase(const ModuleBase&) = default;
    ModuleBase(ModuleBase&&) = default;
    ModuleBase& operator=(const ModuleBase&) = default;
    ModuleBase& operator=(ModuleBase&&) = default;

    [[nodiscard]] const std::string& get_name() const;
    [[nodiscard]] NamedParameters parameters() const;

    void set_train_mode(TrainMode mode);
    [[nodiscard]] TrainMode get_train_mode() const;
};

}  // namespace ttml::autograd
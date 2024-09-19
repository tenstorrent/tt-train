#pragma once
#include <memory>
#include <optional>

#include "autograd/graph.hpp"
#include "core/ttnn_all_includes.hpp"
#include "graph.hpp"

namespace ttml::autograd {
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    tt::tt_metal::Tensor m_value;
    tt::tt_metal::Tensor m_grad;
    bool m_require_grad = true;
    std::optional<NodeId> m_node_id;

public:
    Tensor() = default;
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;
    explicit Tensor(tt::tt_metal::Tensor m_value, bool require_grad = true);
    ~Tensor() = default;

    void set_value(const tt::tt_metal::Tensor &value) { m_value = value; }
    void set_grad(const tt::tt_metal::Tensor &grad) { m_grad = grad; }
    void set_node(const std::optional<NodeId> &node);
    void clean_node() { m_node_id = std::nullopt; }
    void add_grad(const tt::tt_metal::Tensor &grad);
    void set_require_grad(bool require_grad) { m_require_grad = require_grad; }

    const tt::tt_metal::Tensor &get_value() const { return m_value; }
    const tt::tt_metal::Tensor &get_grad() const { return m_grad; }
    tt::tt_metal::Tensor &get_grad() { return m_grad; }
    bool get_require_grad() const { return m_require_grad; }
    const std::optional<NodeId> &get_node() const { return m_node_id; }

    void backward();

    bool is_grad_initialized() const;

private:
    void try_init_grad(bool init_ones = false);
};

using TensorPtr = std::shared_ptr<Tensor>;

// TODO: In future implement create tensor without variadic templates to help with code hints in IDE
template <typename... Args>
TensorPtr create_tensor(Args &&... args) {
    return std::make_shared<Tensor>(std::forward<Args>(args)...);
}

}  // namespace ttml::autograd
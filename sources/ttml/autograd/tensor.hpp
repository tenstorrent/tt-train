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
    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) noexcept = default;
    Tensor(tt::tt_metal::Tensor m_value, tt::tt_metal::Tensor m_grad);
    ~Tensor() = default;

    void set_value(const tt::tt_metal::Tensor &value) { m_value = value; }
    void set_grad(const tt::tt_metal::Tensor &grad) { m_grad = grad; }
    void set_node(const std::optional<NodeId> &node) { m_node_id = node; }
    void clean_node() { m_node_id = std::nullopt; }
    void add_grad(const tt::tt_metal::Tensor &grad);
    void set_require_grad(bool require_grad) { m_require_grad = require_grad; }

    const tt::tt_metal::Tensor &get_value() const { return m_value; }
    const tt::tt_metal::Tensor &get_grad() const { return m_grad; }
    tt::tt_metal::Tensor &get_grad() { return m_grad; }
    bool get_require_grad() const { return m_require_grad; }
    const std::optional<NodeId> &get_node() const { return m_node_id; }
};

using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace ttml::autograd
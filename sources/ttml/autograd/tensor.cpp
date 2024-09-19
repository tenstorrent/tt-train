#include "tensor.hpp"

#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

// TODO: implement stack based topological sort
void topological_sort(
    size_t node_id,
    const std::vector<std::vector<size_t>>& edges,
    std::unordered_set<size_t>& visited,
    std::vector<size_t>& sorted_nodes) {
    if (visited.contains(node_id)) {
        return;
    }
    visited.insert(node_id);
    for (const auto& next_node : edges[node_id]) {
        topological_sort(next_node, edges, visited, sorted_nodes);
    }
    sorted_nodes.push_back(node_id);
}

}  // namespace

namespace ttml::autograd {

Tensor::Tensor(tt::tt_metal::Tensor m_value, bool require_grad) :
    m_value(std::move(m_value)), m_require_grad(require_grad) {}

void Tensor::add_grad(const tt::tt_metal::Tensor& grad) {
    try_init_grad();

    const auto& grad_shape = grad.get_shape();
    const auto& m_grad_shape = m_grad.get_shape();
    if (grad_shape != m_grad_shape) {
        throw std::logic_error(
            fmt::format("Shapes of gradients are not equal. Expected: {}, got: {}", m_grad_shape, grad_shape));
    }

    m_grad = ttnn::add(m_grad, grad);
}

void Tensor::backward() {
    if (!m_node_id.has_value()) {
        return;
    }
    std::vector<size_t> sorted_nodes;
    std::unordered_set<std::size_t> visited_nodes;
    const auto& graph = m_node_id->get_graph();
    topological_sort(m_node_id->get_id(), graph.get_edges(), visited_nodes, sorted_nodes);

    const auto& graph_nodes = graph.get_graph_nodes();
    std::ranges::reverse(sorted_nodes);
    try_init_grad(/* init_ones */ true);
    for (const auto& node_id : sorted_nodes) {
        graph_nodes[node_id].grad_function();
    }
}

bool Tensor::is_grad_initialized() const { return core::is_tensor_initialized(get_grad()); }

void Tensor::try_init_grad(bool init_ones) {
    if (is_grad_initialized()) {
        return;
    }
    this->set_grad(init_ones ? ttml::core::ones_like(m_value) : ttml::core::zeros_like(m_value));
}
void Tensor::set_node(const std::optional<NodeId>& node) {
    if (m_node_id.has_value()) {
        throw std::runtime_error("Graph node is already set for this tensor!");
    }
    m_node_id = node;
}
}  // namespace ttml::autograd
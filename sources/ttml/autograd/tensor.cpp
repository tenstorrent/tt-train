#include "tensor.hpp"

#include "core/not_null.hpp"

namespace ttml::autograd {

Tensor::Tensor(tt::tt_metal::Tensor m_value, tt::tt_metal::Tensor m_grad) :
    m_value(std::move(m_value)), m_grad(std::move(m_grad)) {}

void Tensor::add_grad(const tt::tt_metal::Tensor& grad) { m_grad = ttnn::add_(m_grad, grad); }

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
    for (const auto& v : edges[node_id]) {
        topological_sort(v, edges, visited, sorted_nodes);
    }
    sorted_nodes.push_back(node_id);
}

}  // namespace

void Tensor::backward() {
    if (!m_node_id.has_value()) {
        // TODO: throw exception?
        return;
    }

    std::vector<size_t> sorted_nodes;
    std::unordered_set<std::size_t> visited_nodes;
    auto graph_ptr = m_node_id->get_graph();
    topological_sort(m_node_id->get_id(), graph_ptr->get_edges(), visited_nodes, sorted_nodes);

    const auto& graph_nodes = graph_ptr->get_graph_nodes();
    std::ranges::reverse(sorted_nodes);
    for (const auto& node_id : sorted_nodes) {
        graph_nodes[node_id].grad_function();
    }
}

}  // namespace ttml::autograd
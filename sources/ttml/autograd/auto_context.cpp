#include "auto_context.hpp"

#include <optional>

namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() { return m_generator; }

void AutoContext::set_seed(uint32_t seed) {
    m_seed = seed;
    m_generator = std::mt19937(m_seed);
}

AutoContext& AutoContext::get_instance() {
    static AutoContext instance;
    return instance;
}
std::optional<NodeId> AutoContext::add_backward_node(GradFunction&& grad_function, std::span<NodeId> links) {
    if (m_grads_mode == GradMode::DISABLED) {
        return std::nullopt;
    }
    return m_graph.add_node(std::move(grad_function), links);
}
void AutoContext::set_gradient_mode(GradMode mode) { m_grads_mode = mode; }
GradMode AutoContext::get_gradient_mode() const { return m_grads_mode; }

uint32_t AutoContext::generate_module_id() { return module_counter++; }

}  // namespace ttml::autograd
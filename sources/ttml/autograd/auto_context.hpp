#pragma once

#include <random>

#include "graph.hpp"

namespace ttml::autograd {

enum class GradMode { ENABLED, DISABLED };

class AutoContext {
public:
    // Delete copy constructor and assignment operator to prevent copying
    AutoContext(const AutoContext&) = delete;
    AutoContext& operator=(const AutoContext&) = delete;
    AutoContext(AutoContext&&) = delete;
    AutoContext& operator=(AutoContext&&) = delete;
    // Static method to access the singleton instance
    static AutoContext& get_instance();

    std::mt19937& get_generator();

    void set_seed(uint32_t seed);

    [[nodiscard]] uint32_t get_seed() const;

    std::optional<NodeId> add_backward_node(GradFunction&& grad_function, std::span<NodeId> links);

    void set_gradient_mode(GradMode mode);
    [[nodiscard]] GradMode get_gradient_mode() const;

    ~AutoContext() = default;  // to make it work with unique_ptr.

    [[nodiscard]] uint32_t generate_module_id();

private:
    AutoContext() = default;

    std::mt19937 m_generator;
    uint32_t m_seed = 5489U;

    GradMode m_grads_mode = GradMode::ENABLED;

    Graph m_graph;

    uint32_t module_counter = 0;
};

inline auto& ctx() { return ttml::autograd::AutoContext::get_instance(); }
}  // namespace ttml::autograd
#pragma once

#include <random>

#include "graph.hpp"
namespace ttml::autograd {

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

    void set_seed(unsigned int seed);

    [[nodiscard]] unsigned int get_seed() const;

    ~AutoContext() = default;  // to make it work with unique_ptr.

   private:
    AutoContext() = default;

    std::mt19937 m_generator;
    unsigned int m_seed = 5489U;

    Graph graph;
};

}  // namespace ttml::autograd
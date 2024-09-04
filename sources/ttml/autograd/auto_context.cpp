#include "auto_context.hpp"
namespace ttml::autograd {

std::mt19937& AutoContext::get_generator() { return m_generator; }

void AutoContext::set_seed(unsigned int seed) {
    m_seed = seed;
    m_generator = std::mt19937(m_seed);
}

AutoContext& AutoContext::get_instance() {
    static AutoContext instance;
    return instance;
}
}  // namespace ttml::autograd
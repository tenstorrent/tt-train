#include "utils.hpp"

void LossAverageMeter::update(float loss, size_t count) {
    m_sum += loss * static_cast<float>(count);
    m_count += count;
}

float LossAverageMeter::average() const {
    if (m_count == 0) {
        return 0.F;
    }
    return m_sum / static_cast<float>(m_count);
}

void LossAverageMeter::reset() {
    m_sum = 0.0F;
    m_count = 0;
}

std::string read_file_to_str(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

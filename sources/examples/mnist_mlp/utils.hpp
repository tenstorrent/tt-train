#include <cstddef>

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1) {
        m_sum += loss * static_cast<float>(count);
        m_count += count;
    }

    [[nodiscard]] float average() const {
        if (m_count == 0) {
            return 0.F;
        }
        return m_sum / static_cast<float>(m_count);
    }

    void reset() {
        m_sum = 0.0F;
        m_count = 0;
    }
};

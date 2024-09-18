#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>

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

class Timers {
public:
    void start(const std::string& name) { m_timers[name] = std::chrono::high_resolution_clock::now(); }

    auto stop(const std::string& name) {
        assert(m_timers.contains(name));
        auto start_time = m_timers[name];
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return duration;
    }

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_timers;
};
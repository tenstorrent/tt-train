#pragma once

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
    void update(float loss, size_t count = 1);

    [[nodiscard]] float average() const;

    void reset();
};

class Timers {
public:
    void start(const std::string_view& name);

    long long stop(const std::string_view& name);

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_timers;
};
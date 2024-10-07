#include <fstream>
#include <iostream>
#include <sstream>

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1);

    [[nodiscard]] float average() const;

    void reset();
};

std::string read_file_to_str(const std::string& file_path);

#include <iostream>

#include "core/ttnn_all_includes.hpp"

std::string read_file_to_str(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    const std::string data_folder = "/home/ubuntu/ML-Framework-CPP/sources/examples/nano_gpt/data";
    const std::string data_path = data_folder + "/shakespeare.txt";

    std::string text;
    try {
        text = read_file_to_str(data_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Text length: " << text.size() << std::endl;

    return 0;
}
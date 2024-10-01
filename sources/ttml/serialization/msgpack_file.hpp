#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ttml::serialization {

class MsgPackFile {
public:
    MsgPackFile();
    ~MsgPackFile();

    // Copy constructor
    MsgPackFile(const MsgPackFile& other) = delete;

    // Copy assignment operator
    MsgPackFile& operator=(const MsgPackFile& other) = delete;

    // Move constructor
    MsgPackFile(MsgPackFile&& other) noexcept;

    // Move assignment operator
    MsgPackFile& operator=(MsgPackFile&& other) = delete;

    // Methods to put different types
    void put(std::string_view key, int value);
    void put(std::string_view key, float value);
    void put(std::string_view key, double value);
    void put(std::string_view key, uint32_t value);
    void put(std::string_view key, std::string_view value);

    // Overloads for std::span
    void put(std::string_view key, std::span<const int> value);
    void put(std::string_view key, std::span<const float> value);
    void put(std::string_view key, std::span<const double> value);
    void put(std::string_view key, std::span<const uint32_t> value);
    void put(std::string_view key, std::span<const std::string> value);

    // Serialization method
    void serialize(const std::string& filename);

    // Deserialization method
    void deserialize(const std::string& filename);

    // Methods to get values
    bool get(std::string_view key, int& value) const;
    bool get(std::string_view key, float& value) const;
    bool get(std::string_view key, double& value) const;
    bool get(std::string_view key, uint32_t& value) const;
    bool get(std::string_view key, std::string& value) const;

    // Methods to get vectors (from spans)
    bool get(std::string_view key, std::vector<int>& value) const;
    bool get(std::string_view key, std::vector<float>& value) const;
    bool get(std::string_view key, std::vector<double>& value) const;
    bool get(std::string_view key, std::vector<uint32_t>& value) const;
    bool get(std::string_view key, std::vector<std::string>& value) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace ttml::serialization
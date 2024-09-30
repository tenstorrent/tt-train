#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>
namespace ttml::serialization {

class HDF5File {
public:
    ~HDF5File();

    HDF5File(const HDF5File&) = delete;
    HDF5File& operator=(const HDF5File&) = delete;

    HDF5File(HDF5File&&) noexcept;
    HDF5File& operator=(HDF5File&&) noexcept;

    HDF5File(const std::string& filename, bool read_only);

    template <class T>
    void create_storage(const std::string& storage_name, std::span<unsigned long long> dims);

    template <class T>
    void write_storage(const std::string& storage_name, std::span<T> data);

    template <class T>
    std::vector<T> read_storage(const std::string& storage_name);

    template <class T>
    void write_attribute(const std::string& storage_name, const std::string& attr_name, const T& attr);

    template <class T>
    void write_attribute_vec(const std::string& storage_name, const std::string& attr_name, std::span<T> attr);

    template <class T>
    T read_attribute(const std::string& storage_name, const std::string& attr_name);

    template <class T>
    std::vector<T> read_attribute_vec(const std::string& storage_name, const std::string& attr_name);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace ttml::serialization
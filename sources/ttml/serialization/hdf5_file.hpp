#pragma once

#include <memory>
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

    HDF5File(const std::string& filename, bool readOnly);

    template <class T>
    void create_dataset(const std::string& dataset_name, const std::vector<unsigned long long>& dims);

    template <class T>
    void write_dataset(const std::string& dataset_name, const std::vector<T>& data);

    template <class T>
    std::vector<T> read_dataset(const std::string& dataset_name);

    template <class T>
    void write_attribute(const std::string& dataset_name, const std::string& attr_name, const T& attr);

    template <class T>
    void write_attribute_vec(const std::string& dataset_name, const std::string& attr_name, const std::vector<T>& attr);

    template <class T>
    T read_attribute(const std::string& dataset_name, const std::string& attr_name);

    template <class T>
    std::vector<T> read_attribute_vec(const std::string& dataset_name, const std::string& attr_name);

private:
    // Forward declaration of the implementation class
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace ttml::serialization
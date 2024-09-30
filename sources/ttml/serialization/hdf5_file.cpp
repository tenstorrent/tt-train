
// HDF5File.cpp
#include "hdf5_file.hpp"

#include <H5public.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "H5Cpp.h"

using namespace H5;

namespace {
template <typename TFull>
const H5::PredType& getHDF5PredType() {
    using T = std::remove_cv_t<std::remove_reference_t<TFull>>;
    if constexpr (std::is_same_v<T, int>) {
        return H5::PredType::NATIVE_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return H5::PredType::NATIVE_UINT;
    } else if constexpr (std::is_same_v<T, float>) {
        return H5::PredType::NATIVE_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        return H5::PredType::NATIVE_DOUBLE;
    } else if constexpr (std::is_same_v<T, long>) {
        return H5::PredType::NATIVE_LONG;
    } else if constexpr (std::is_same_v<T, short>) {
        return H5::PredType::NATIVE_SHORT;
    } else if constexpr (std::is_same_v<T, char>) {
        return H5::PredType::NATIVE_CHAR;
    } else if constexpr (std::is_same_v<T, std::string_view>) {
        return H5::PredType::NATIVE_CHAR;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return H5::PredType::NATIVE_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        return H5::PredType::NATIVE_UCHAR;
    } else if constexpr (std::is_same_v<T, long long>) {
        return H5::PredType::NATIVE_LLONG;
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return H5::PredType::NATIVE_ULLONG;
    } else if constexpr (std::is_same_v<T, bool>) {
        return H5::PredType::NATIVE_HBOOL;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for HDF5 PredType mapping.");
    }
}
}  // namespace

namespace ttml::serialization {
// Implementation class
class HDF5File::Impl {
private:
    // File operations
    void create_file(const std::string& filename) { m_file = std::make_unique<H5File>(filename, H5F_ACC_TRUNC); }

    void open_file(const std::string& filename, bool read_only) {
        auto flag = read_only ? H5F_ACC_RDONLY : H5F_ACC_RDWR;

        m_file = std::make_unique<H5File>(filename, flag);
    }

public:
    Impl(const std::string& filename, bool read_only) { open_file(filename, read_only); }

    template <class T>
    void create_storage(const std::string& storage_name, std::span<hsize_t> dims) {
        DataSpace dataspace((int)dims.size(), dims.data());

        m_file->createDataSet(storage_name, getHDF5PredType<T>(), dataspace);
    }

    template <class T>
    void write_storage(const std::string& storage_name, std::span<T> data) {
        DataSet dataset = m_file->openDataSet(storage_name);
        dataset.write(data.data(), getHDF5PredType<T>());
    }

    template <class T>
    std::vector<T> read_storage(const std::string& storage_name) {
        DataSet dataset = m_file->openDataSet(storage_name);
        DataSpace dataspace = dataset.getSpace();

        hsize_t dims = 0;
        dataspace.getSimpleExtentDims(&dims, nullptr);
        std::vector<T> data(dims);
        dataset.read(data.data(), getHDF5PredType<T>());
        return data;
    }

    template <class T>
    void write_attribute_vec(const std::string& storage_name, const std::string& attr_name, std::span<T> data) {
        DataSet dataset = m_file->openDataSet(storage_name);
        hsize_t dims = data.size();
        DataSpace attr_space(1, &dims);
        auto attr_type = getHDF5PredType<T>();
        Attribute attr = dataset.createAttribute(attr_name, attr_type, attr_space);
        attr.write(attr_type, data.data());
    }
    template <class T>
    void write_attribute(const std::string& storage_name, const std::string& attr_name, const T& value) {
        DataSet dataset = m_file->openDataSet(storage_name);
        StrType attr_type(getHDF5PredType<T>(), H5T_VARIABLE);
        Attribute attr = dataset.createAttribute(attr_name, attr_type, DataSpace());
        attr.write(attr_type, &value);
    }

    void write_attribute_string(
        const std::string& storage_name, const std::string& attr_name, const std::string& attr) {
        DataSet dataset = m_file->openDataSet(storage_name);

        // Define variable-length string type
        StrType str_type(PredType::C_S1, H5T_VARIABLE);

        // Create the attribute
        Attribute attribute = dataset.createAttribute(attr_name, str_type, DataSpace(H5S_SCALAR));

        // Write the attribute
        attribute.write(str_type, attr);
    }

    template <class T>
    std::vector<T> read_attribute_vec(const std::string& storage_name, const std::string& attr_name) {
        DataSet dataset = m_file->openDataSet(storage_name);
        if (!dataset.attrExists(attr_name)) {
            throw std::runtime_error("Attribute " + attr_name + " does not exist.");
        }

        Attribute attr = dataset.openAttribute(attr_name);
        DataSpace attrSpace = attr.getSpace();
        int rank = attrSpace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        attrSpace.getSimpleExtentDims(dims.data(), nullptr);
        std::vector<T> output(dims[0]);
        attr.read(getHDF5PredType<T>(), output.data());
        return output;
    }

    std::string read_attribute_string(const std::string& storage_name, const std::string& attr_name) {
        DataSet dataset = m_file->openDataSet(storage_name);
        if (!dataset.attrExists(attr_name)) {
            throw std::runtime_error("Attribute " + attr_name + " does not exist.");
        }
        Attribute attr = dataset.openAttribute(attr_name);

        StrType dtype = attr.getStrType();
        std::string output;
        attr.read(dtype, &output);
        return output;
    }

    template <class T>
    T read_attribute(const std::string& storage_name, const std::string& attr_name) {
        DataSet dataset = m_file->openDataSet(storage_name);
        if (!dataset.attrExists(attr_name)) {
            throw std::runtime_error("Attribute " + attr_name + " does not exist.");
        }
        Attribute attr = dataset.openAttribute(attr_name);

        T output;
        H5::PredType mem_type = getHDF5PredType<T>();
        attr.read(mem_type, &output);
        return output;
    }

private:
    std::unique_ptr<H5File> m_file;
};

// HDF5File Method Implementations

HDF5File::HDF5File(const std::string& filename, bool read_only) : pImpl(std::make_unique<Impl>(filename, read_only)) {}
HDF5File::~HDF5File() = default;

HDF5File::HDF5File(HDF5File&& other) noexcept = default;

HDF5File& HDF5File::operator=(HDF5File&& other) noexcept = default;

template <>
void HDF5File::create_storage<float>(const std::string& storage_name, std::span<unsigned long long> dims) {
    pImpl->create_storage<float>(storage_name, dims);
}
template <>
void HDF5File::create_storage<uint32_t>(const std::string& storage_name, std::span<unsigned long long> dims) {
    pImpl->create_storage<uint32_t>(storage_name, dims);
}

template <>
void HDF5File::create_storage<int>(const std::string& storage_name, std::span<unsigned long long> dims) {
    pImpl->create_storage<int>(storage_name, dims);
}

template <>
void HDF5File::write_storage<float>(const std::string& storage_name, std::span<float> data) {
    pImpl->write_storage(storage_name, data);
}

template <>
void HDF5File::write_storage<uint32_t>(const std::string& storage_name, std::span<uint32_t> data) {
    pImpl->write_storage(storage_name, data);
}

template <>
void HDF5File::write_storage<int>(const std::string& storage_name, std::span<int> data) {
    pImpl->write_storage(storage_name, data);
}

template <>
std::vector<float> HDF5File::read_storage<float>(const std::string& storage_name) {
    return pImpl->read_storage<float>(storage_name);
}

template <>
std::vector<uint32_t> HDF5File::read_storage<uint32_t>(const std::string& storage_name) {
    return pImpl->read_storage<uint32_t>(storage_name);
}

template <>
std::vector<int> HDF5File::read_storage<int>(const std::string& storage_name) {
    return pImpl->read_storage<int>(storage_name);
}

template <>
void HDF5File::write_attribute<std::string>(
    const std::string& storage_name, const std::string& attr_name, const std::string& attr) {
    pImpl->write_attribute_string(storage_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute<uint32_t>(
    const std::string& storage_name, const std::string& attr_name, const uint32_t& attr) {
    pImpl->write_attribute<uint32_t>(storage_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute<int>(const std::string& storage_name, const std::string& attr_name, const int& attr) {
    pImpl->write_attribute<int>(storage_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute_vec<float>(
    const std::string& storage_name, const std::string& attr_name, std::span<float> attr) {
    pImpl->write_attribute_vec<float>(storage_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute_vec<int>(
    const std::string& storage_name, const std::string& attr_name, std::span<int> attr) {
    pImpl->write_attribute_vec<int>(storage_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute_vec<uint32_t>(
    const std::string& storage_name, const std::string& attr_name, std::span<uint32_t> attr) {
    pImpl->write_attribute_vec<uint32_t>(storage_name, attr_name, attr);
}

template <>
float HDF5File::read_attribute<float>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute<float>(storage_name, attr_name);
}

template <>
std::string HDF5File::read_attribute<std::string>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute_string(storage_name, attr_name);
}

template <>
uint32_t HDF5File::read_attribute<uint32_t>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute<uint32_t>(storage_name, attr_name);
}

template <>
int HDF5File::read_attribute<int>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute<int>(storage_name, attr_name);
}

template <>
std::vector<float> HDF5File::read_attribute_vec<float>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute_vec<float>(storage_name, attr_name);
}

template <>
std::vector<uint32_t> HDF5File::read_attribute_vec<uint32_t>(
    const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute_vec<uint32_t>(storage_name, attr_name);
}

template <>
std::vector<int> HDF5File::read_attribute_vec<int>(const std::string& storage_name, const std::string& attr_name) {
    return pImpl->read_attribute_vec<int>(storage_name, attr_name);
}

}  // namespace ttml::serialization
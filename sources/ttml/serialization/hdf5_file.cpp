
// HDF5File.cpp
#include "hdf5_file.hpp"

#include <H5public.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "H5Cpp.h"

using namespace H5;

namespace {
template <typename T>
constexpr const H5::PredType& getHDF5PredType() {
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
    void create_file(const std::string& filename) {
        if (m_file && m_file->getId() >= 0) {
            throw std::runtime_error("A m_file is already open.");
        }

        m_file = std::make_unique<H5File>(filename, H5F_ACC_TRUNC);
    }

    void open_file(const std::string& filename, bool readOnly) {
        auto flag = readOnly ? H5F_ACC_RDONLY : H5F_ACC_RDWR;

        m_file = std::make_unique<H5File>(filename, flag);
    }

public:
    Impl(const std::string& filename, bool readOnly) { open_file(filename, readOnly); }

    ~Impl() = default;
    template <class T>
    void create_dataset(const std::string& dataset_name, const std::vector<hsize_t>& dims) {
        DataSpace dataspace((int)dims.size(), dims.data());

        m_file->createDataSet(dataset_name, getHDF5PredType<T>(), dataspace);
    }

    template <class T>
    void write_dataset(const std::string& dataset_name, const std::vector<T>& data) {
        DataSet dataset = m_file->openDataSet(dataset_name);
        dataset.write(data.data(), getHDF5PredType<T>());
    }

    template <class T>
    std::vector<T> read_dataset(const std::string& dataset_name) {
        DataSet dataset = m_file->openDataSet(dataset_name);
        DataSpace dataspace = dataset.getSpace();

        hsize_t dims = 0;
        dataspace.getSimpleExtentDims(&dims, nullptr);
        std::vector<T> data(dims);
        dataset.read(data.data(), getHDF5PredType<T>());
        return data;
    }

    template <class T>
    void write_attribute_vec(
        const std::string& dataset_name, const std::string& attr_name, const std::vector<T>& data) {
        DataSet dataset = m_file->openDataSet(dataset_name);
        hsize_t dims = data.size();
        DataSpace attr_space(1, &dims);
        auto attr_type = getHDF5PredType<T>();
        Attribute attr = dataset.createAttribute(attr_name, attr_type, attr_space);
        attr.write(attr_type, data.data());
    }
    template <class T>
    void write_attribute(const std::string& dataset_name, const std::string& attr_name, const T& value) {
        DataSet dataset = m_file->openDataSet(dataset_name);
        StrType attr_type(getHDF5PredType<T>(), H5T_VARIABLE);
        Attribute attr = dataset.createAttribute(attr_name, attr_type, DataSpace());
        attr.write(attr_type, &value);
    }

    template <class T>
    std::vector<T> read_attribute_vec(const std::string& dataset_name, const std::string& attr_name) {
        DataSet dataset = m_file->openDataSet(dataset_name);
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

    template <class T>
    T read_attribute(const std::string& dataset_name, const std::string& attr_name) {
        DataSet dataset = m_file->openDataSet(dataset_name);
        if (!dataset.attrExists(attr_name)) {
            throw std::runtime_error("Attribute " + attr_name + " does not exist.");
        }
        Attribute attr = dataset.openAttribute(attr_name);

        StrType dtype = attr.getStrType();
        T output;
        attr.read(dtype, &output);
        return output;
    }

private:
    std::unique_ptr<H5File> m_file;
};

// HDF5File Method Implementations

HDF5File::HDF5File(const std::string& filename, bool readOnly) : pImpl(std::make_unique<Impl>(filename, readOnly)) {}
HDF5File::~HDF5File() = default;

HDF5File::HDF5File(HDF5File&& other) noexcept = default;

HDF5File& HDF5File::operator=(HDF5File&& other) noexcept = default;

template <>
void HDF5File::create_dataset<float>(const std::string& dataset_name, const std::vector<unsigned long long>& dims) {
    pImpl->create_dataset<float>(dataset_name, dims);
}
template <>
void HDF5File::create_dataset<uint32_t>(const std::string& dataset_name, const std::vector<unsigned long long>& dims) {
    pImpl->create_dataset<uint32_t>(dataset_name, dims);
}

template <>
void HDF5File::write_dataset<float>(const std::string& dataset_name, const std::vector<float>& data) {
    pImpl->write_dataset(dataset_name, data);
}

template <>
void HDF5File::write_dataset<uint32_t>(const std::string& dataset_name, const std::vector<uint32_t>& data) {
    pImpl->write_dataset(dataset_name, data);
}

template <>
std::vector<float> HDF5File::read_dataset<float>(const std::string& dataset_name) {
    return pImpl->read_dataset<float>(dataset_name);
}

template <>
std::vector<uint32_t> HDF5File::read_dataset<uint32_t>(const std::string& dataset_name) {
    return pImpl->read_dataset<uint32_t>(dataset_name);
}

template <>
void HDF5File::write_attribute<std::string>(
    const std::string& dataset_name, const std::string& attr_name, const std::string& attr) {
    pImpl->write_attribute<std::string>(dataset_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute<uint32_t>(
    const std::string& dataset_name, const std::string& attr_name, const uint32_t& attr) {
    pImpl->write_attribute<uint32_t>(dataset_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute_vec<float>(
    const std::string& dataset_name, const std::string& attr_name, const std::vector<float>& attr) {
    pImpl->write_attribute_vec<float>(dataset_name, attr_name, attr);
}

template <>
void HDF5File::write_attribute_vec<uint32_t>(
    const std::string& dataset_name, const std::string& attr_name, const std::vector<uint32_t>& attr) {
    pImpl->write_attribute_vec<uint32_t>(dataset_name, attr_name, attr);
}

template <>
float HDF5File::read_attribute<float>(const std::string& dataset_name, const std::string& attr_name) {
    return pImpl->read_attribute<float>(dataset_name, attr_name);
}

template <>
uint32_t HDF5File::read_attribute<uint32_t>(const std::string& dataset_name, const std::string& attr_name) {
    return pImpl->read_attribute<uint32_t>(dataset_name, attr_name);
}

template <>
std::vector<float> HDF5File::read_attribute_vec<float>(const std::string& dataset_name, const std::string& attr_name) {
    return pImpl->read_attribute_vec<float>(dataset_name, attr_name);
}

template <>
std::vector<uint32_t> HDF5File::read_attribute_vec<uint32_t>(
    const std::string& dataset_name, const std::string& attr_name) {
    return pImpl->read_attribute_vec<uint32_t>(dataset_name, attr_name);
}

}  // namespace ttml::serialization
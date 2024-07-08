#include "fourier_mellin.hpp"

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef MODULE_NAME
#error "MODULE_NAME is not defined"
#endif

namespace py = pybind11;

template<unsigned int Channels>
cv::Mat numpy_to_mat(py::array_t<float> input) {
    // TODO: Make this properly, but for some reason this is necessary to manually set channels
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_32FC(Channels), (float*)buf.ptr);
    return mat;
}

py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    py::ssize_t rows = mat.rows;
    py::ssize_t cols = mat.cols;
    py::ssize_t channels = mat.channels();
    py::ssize_t total_size = rows * cols * channels;

    std::vector<float> data(total_size);
    if (mat.isContinuous()) {
        std::memcpy(data.data(), mat.ptr<float>(), total_size * sizeof(float));
    } else {
        for (int i = 0; i < rows; ++i) {
            std::memcpy(data.data() + i * cols * channels, mat.ptr<float>(i), cols * channels * sizeof(float));
        }
    }

    return py::array_t<float>(
        {rows, cols, channels}, // shape
        {cols * channels * sizeof(float), channels * sizeof(float), sizeof(float)}, // strides
        data.data() // the data pointer
    );
}

struct PyLogPolarMap{
    int logPolarSize;
    double logBase;
    py::array_t<float> xMap;
    py::array_t<float> yMap;

    static PyLogPolarMap ConvertFromLogPolarMap(const LogPolarMap& polarMap){
        return PyLogPolarMap{
            .logPolarSize = polarMap.logPolarSize,
            .logBase = polarMap.logBase,
            .xMap = mat_to_numpy(polarMap.xMap),
            .yMap = mat_to_numpy(polarMap.yMap),
        };
    }

    LogPolarMap ConvertToLogPolarMap(){
        return LogPolarMap{
            .logPolarSize = logPolarSize,
            .logBase = logBase,
            .xMap = numpy_to_mat<1>(xMap),
            .yMap = numpy_to_mat<1>(yMap),
        };
    }
};

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def_readwrite("x_offset", &Transform::xOffset)
        .def_readwrite("y_offset", &Transform::yOffset)
        .def_readwrite("scale", &Transform::scale)
        .def_readwrite("rotation", &Transform::rotation)
        .def_readwrite("response", &Transform::response);

    py::class_<PyLogPolarMap>(m, "LogPolarMap")
        .def(py::init<>())
        .def_readwrite("log_polar_size", &PyLogPolarMap::logPolarSize)
        .def_readwrite("log_base", &PyLogPolarMap::logBase)
        .def_readwrite("x_map", &PyLogPolarMap::xMap)
        .def_readwrite("y_map", &PyLogPolarMap::yMap);

    m.def("get_filters", [](int cols, int rows) -> auto {
        auto highPassFilter = getHighPassFilter(rows, cols);
        auto apodizationWindow = getApodizationWindow(cols, rows, std::min(rows, cols));

        auto highPassFilter2 = mat_to_numpy(highPassFilter);
        auto apodizationWindow2 = mat_to_numpy(apodizationWindow);

        return std::make_tuple(highPassFilter2, apodizationWindow2);
    }, "Do something");

    m.def("create_log_polar_map", [](int cols, int rows) -> auto {
        auto polarMap = createLogPolarMap(cols, rows);
        return PyLogPolarMap::ConvertFromLogPolarMap(polarMap);
    }, "Do something");

    m.def("process_image", [](py::array_t<float> img, py::array_t<float> highPassFilter, py::array_t<float> apodizationWindow, PyLogPolarMap logPolarMap){
        auto logPolarMap2 = logPolarMap.ConvertToLogPolarMap();
        auto img2 = numpy_to_mat<1>(img);
        auto highPassFilter2 = numpy_to_mat<2>(highPassFilter);
        auto apodizationWindow2 = numpy_to_mat<1>(apodizationWindow);
        auto logPolarImg = getProcessedImage(img2, highPassFilter2, apodizationWindow2, logPolarMap2);
        return mat_to_numpy(logPolarImg);
    }, "Process Image");

    m.def("register_image", [](py::array_t<float> img0, py::array_t<float> img1, py::array_t<float>logPolar0, py::array_t<float> logPolar1, PyLogPolarMap logPolarMap){
        auto mat0 = numpy_to_mat<1>(img0);
        auto mat1 = numpy_to_mat<1>(img1);
        auto matLogPolar0 = numpy_to_mat<1>(logPolar0);
        auto matLogPolar1 = numpy_to_mat<1>(logPolar1);
        auto logPolarMap2 = logPolarMap.ConvertToLogPolarMap();
        return registerGrayImage(mat0, mat1, matLogPolar0, matLogPolar1, logPolarMap2);
    }, "Register Images");

    m.def("get_transformed", [](py::array_t<float> img, Transform transform){
        auto mat = numpy_to_mat<3>(img);
        auto transformed = getTransformed(mat, transform);
        return mat_to_numpy(transformed);
    }, "Process Image");

}

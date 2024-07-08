#include "filters.hpp"
#include <numbers>

constexpr long double pi = std::numbers::pi_v<long double>;

cv::Mat linspace(float min, float max, size_t count){
    cv::Mat result = cv::Mat::zeros(count, 1, CV_64F);
    float step = (max - min) / (count - 1);
    for(size_t i=0; i<count; i++) {
        result.at<double>(i, 0) = min + step * i;
    }
    return result;
}

cv::Mat getHighPassFilter(int rows, int cols) {
    cv::Mat y = linspace(-pi / 2.0, pi / 2.0, rows);
    cv::Mat x = linspace(-pi / 2.0, pi / 2.0, cols).t();
    cv::Mat yMat = cv::repeat(y, 1, cols);
    cv::Mat xMat = cv::repeat(x, rows, 1);

    cv::Mat temp = yMat.mul(yMat) + xMat.mul(xMat);
    cv::sqrt(temp, temp);

    cv::Mat filter = cv::Mat(temp.size(), temp.type());
    for(int i=0; i<temp.rows; i++){
        for(int j=0; j<temp.cols; j++){
            filter.at<double>(i, j) = std::cos(temp.at<double>(i, j));
        }
    }

    filter = filter.mul(filter);
    filter = -filter + 1.0;

    cv::Mat channels[] = {filter, filter};
    cv::Mat filterTwoChannel;
    cv::merge(channels, 2, filterTwoChannel);

    cv::Mat filterConverted;
    filterTwoChannel.convertTo(filterConverted, CV_32F);
    return filterConverted;
}

cv::Mat getApodizationWindow(int cols, int rows, int radius){
    cv::Mat hanningWindow;
    cv::createHanningWindow(hanningWindow, cv::Size(radius, radius), CV_32F);
    cv::resize(hanningWindow, hanningWindow, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_CUBIC);
    return hanningWindow;
}

cv::Mat getFilteredImage(const cv::Mat &gray, const cv::Mat& apodizationWindow, const cv::Mat& highPassFilter){
    cv::Mat apodized = gray.mul(apodizationWindow);
    cv::Mat dftResult = fft(apodized);
    cv::Mat filtered = fftShift(dftResult);
    cv::multiply(filtered, highPassFilter, filtered);
    return filtered;
}

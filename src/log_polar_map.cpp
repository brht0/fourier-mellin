#include "log_polar_map.hpp"

#include <numbers>

constexpr long double pi = std::numbers::pi_v<long double>;

LogPolarMap createLogPolarMap(int cols, int rows){
    int logPolarSize = std::max(cols, rows);
    double logBase = std::exp(std::log(logPolarSize * 1.5 / 2.0) / logPolarSize);
    float ellipse_coefficient = rows / (float)cols;

    cv::Mat xMap(logPolarSize, logPolarSize, CV_32FC1);
    cv::Mat yMap(logPolarSize, logPolarSize, CV_32FC1);

    for(int i=0; i<logPolarSize; i++){
        float angle = -(pi / logPolarSize) * i;
        float cos_angle = std::cos(angle) / ellipse_coefficient;
        float sin_angle = std::sin(angle);

        for(int j=0; j<logPolarSize; j++){
            float scale = std::pow(logBase, j);
            xMap.at<float>(i, j) = scale * cos_angle + cols / 2.0f;
            yMap.at<float>(i, j) = scale * sin_angle + rows / 2.0f;
        }
    }
    return LogPolarMap{
        .logPolarSize=logPolarSize,
        .logBase=logBase,
        .xMap=xMap,
        .yMap=yMap,
    };
}

cv::Mat getLogPolarImage(const cv::Mat& img, const cv::Mat& polarMapX, const cv::Mat& polarMapY){
    std::vector<cv::Mat> planes(2);
    cv::Mat log_polar;
    cv::split(img, planes);
    cv::magnitude(planes[0], planes[1], log_polar);
    cv::remap(log_polar, log_polar, polarMapX, polarMapY, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar());
    return log_polar;
}

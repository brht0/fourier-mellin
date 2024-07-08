#include "fourier_mellin.hpp"

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform){
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(img.cols, img.rows)/2.f, transform.rotation, transform.scale);
    cv::Mat transformed;
    cv::warpAffine(img, transformed, rotationMatrix, img.size());

    cv::Mat translateMatrix = cv::Mat::eye(2, 3, CV_64F);
    translateMatrix.at<double>(0, 2) = -transform.offset[0];
    translateMatrix.at<double>(1, 2) = -transform.offset[1];
    cv::warpAffine(transformed, transformed, translateMatrix, transformed.size());

    return transformed;
}

cv::Mat getProcessedImage(const cv::Mat &img, const cv::Mat& highPassFilter, const cv::Mat& apodizationWindow, const LogPolarMap& logPolarMap){
    auto filtered0 = getFilteredImage(img, apodizationWindow, highPassFilter);
    auto logPolar0 = getLogPolarImage(filtered0, logPolarMap.xMap, logPolarMap.yMap);
    return logPolar0;
}

Transform registerGrayImage(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1, const LogPolarMap& logPolarMap){
    auto[logScale, logRotation] = cv::phaseCorrelate(logPolar1, logPolar0);
    double rotation = -logRotation / logPolarMap.logPolarSize * 180.0;
    double scale = 1.0 / std::pow(logPolarMap.logBase, -logScale);

    const auto center = cv::Point(img0.cols, img0.rows) / 2.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, rotation, scale);
    cv::Mat rotated0;
    cv::warpAffine(img0, rotated0, rotationMatrix, img0.size());

    double response;
    auto[xOffset, yOffset] = cv::phaseCorrelate(img1, rotated0, cv::noArray(), &response);

    return Transform{
        .offset = {xOffset, yOffset},
        .scale = scale,
        .rotation = rotation,
        .response = response
    };
}

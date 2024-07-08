#include "fourier_mellin.hpp"

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform) {
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(img.cols, img.rows)/2.f, transform.rotation, transform.scale);
    cv::Mat transformed(img.size(), img.type());
    cv::warpAffine(img, transformed, rotationMatrix, img.size());

    cv::Mat translateMatrix = cv::Mat::eye(2, 3, CV_64F);
    translateMatrix.at<double>(0, 2) = -transform.xOffset;
    translateMatrix.at<double>(1, 2) = -transform.yOffset;
    cv::warpAffine(transformed, transformed, translateMatrix, transformed.size());

    return transformed;
}

cv::Mat getProcessedImage(const cv::Mat &img, const cv::Mat& highPassFilter, const cv::Mat& apodizationWindow, const LogPolarMap& logPolarMap) {
    auto filtered0 = getFilteredImage(img, apodizationWindow, highPassFilter);
    auto logPolar0 = getLogPolarImage(filtered0, logPolarMap.xMap, logPolarMap.yMap);
    return logPolar0;
}

Transform registerGrayImage(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1, const LogPolarMap& logPolarMap) {
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
        .xOffset = xOffset,
        .yOffset = yOffset,
        .scale = scale,
        .rotation = rotation,
        .response = response
    };
}

FourierMellin::FourierMellin(int cols, int rows):
    cols_(cols), rows_(rows),
    highPassFilter_(getHighPassFilter(rows_, cols_)),
    apodizationWindow_(getApodizationWindow(cols_, rows_, std::min(rows, cols))),
    logPolarMap_(createLogPolarMap(cols_, rows_))
{
}

FourierMellin::~FourierMellin() {
}

cv::Mat FourierMellin::GetProcessImage(const cv::Mat &img) const {
    return getProcessedImage(img, highPassFilter_, apodizationWindow_, logPolarMap_);
}

cv::Mat convertToGrayscale(const cv::Mat& img){
    if(img.channels() == 1){
        return img;
    }
    else if(img.channels() == 3){
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    else{
        throw std::runtime_error("Cannot convert to grayscale with " + std::to_string(img.channels()) + " channels.");
        return cv::Mat();
    }
}

std::tuple<cv::Mat, Transform> FourierMellin::GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1) const {
    cv::Mat gray0 = convertToGrayscale(img0);
    cv::Mat gray1 = convertToGrayscale(img1);

    auto logPolar0 = GetProcessImage(gray0);
    auto logPolar1 = GetProcessImage(gray1);

    auto transform = registerGrayImage(gray0, gray1, logPolar0, logPolar1, logPolarMap_);
    auto transformed = getTransformed(img0, transform);

    return std::make_tuple(transformed, transform);
}

FourierMellinContinuous::FourierMellinContinuous(int cols, int rows):
    cols_(cols), rows_(rows),
    highPassFilter_(getHighPassFilter(rows_, cols_)),
    apodizationWindow_(getApodizationWindow(cols_, rows_, std::min(rows, cols))),
    logPolarMap_(createLogPolarMap(cols_, rows_)),
    isFirst_(true)
{
}

FourierMellinContinuous::~FourierMellinContinuous() {
}

std::tuple<cv::Mat, Transform> FourierMellinContinuous::GetRegisteredImage(const cv::Mat &img){
    cv::Mat gray = convertToGrayscale(img);
    auto logPolar = getProcessedImage(gray, highPassFilter_, apodizationWindow_, logPolarMap_);

    if(std::exchange(isFirst_, false)){
        prevGray_ = gray;
        prevLogPolar_ = logPolar;
        transformSum_ = Transform{};
        return {cv::Mat(), Transform{}};
    }
    else{
        auto transform = registerGrayImage(gray, prevGray_, logPolar, prevLogPolar_, logPolarMap_);
        auto transformed = getTransformed(gray, transform);

        prevGray_ = gray;
        prevLogPolar_ = logPolar;
        if(transformSum_.scale < 1e-5){
            transformSum_ = transform;
        }
        else{
            transformSum_ = transform + transformSum_;
        }
        return {transformed, transformSum_};
    }
}

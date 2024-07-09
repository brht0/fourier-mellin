#include "fourier_mellin.hpp"

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

FourierMellinContinuous::FourierMellinContinuous(int cols, int rows, double edgeCrop):
    cols_(cols), rows_(rows),
    edgeCrop_(edgeCrop),
    highPassFilter_(getHighPassFilter(rows_, cols_)),
    apodizationWindow_(getApodizationWindow(cols_, rows_, std::min(rows, cols))),
    logPolarMap_(createLogPolarMap(cols_, rows_)),
    isFirst_(true)
{
}

FourierMellinContinuous::~FourierMellinContinuous() {
}

std::tuple<cv::Mat, Transform> FourierMellinContinuous::GetRegisteredImage(const cv::Mat &img) {
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

        prevGray_ = gray;
        prevLogPolar_ = logPolar;
        if(transformSum_.scale < 1e-5){
            transformSum_ = transform;
        }
        else{
            transformSum_ = transform + transformSum_;
            transformSum_.xOffset += (- transformSum_.xOffset) * pullToCenterRatio_;
            transformSum_.yOffset += (- transformSum_.yOffset) * pullToCenterRatio_;
        }
        auto transformed = getTransformed(img, transformSum_);
        if(edgeCrop_ != 0.0){
            auto cropped = getCropped(transformed, edgeCrop_ * cols_, edgeCrop_ * rows_, (1.0 - edgeCrop_) * cols_, (1.0 - edgeCrop_) * rows_);
            cv::resize(cropped, cropped, cv::Size(cols_, rows_), cv::INTER_LINEAR);
            return {cropped, transformSum_};
        }
        else{
            return {transformed, transformSum_};
        }
    }
}

FourierMellinWithReference::FourierMellinWithReference(int cols, int rows):
    cols_(cols), rows_(rows),
    highPassFilter_(getHighPassFilter(rows_, cols_)),
    apodizationWindow_(getApodizationWindow(cols_, rows_, std::min(rows, cols))),
    logPolarMap_(createLogPolarMap(cols_, rows_))
{
}

FourierMellinWithReference::~FourierMellinWithReference() {
}

void FourierMellinWithReference::SetReference(const cv::Mat &img) {
    reference_ = convertToGrayscale(img);
    referenceLogPolar_ = getProcessedImage(reference_, highPassFilter_, apodizationWindow_, logPolarMap_);
}

std::tuple<cv::Mat, Transform> FourierMellinWithReference::GetRegisteredImage(const cv::Mat &img) {
    cv::Mat gray = convertToGrayscale(img);
    auto logPolar = getProcessedImage(gray, highPassFilter_, apodizationWindow_, logPolarMap_);

    auto transform = registerGrayImage(gray, reference_, logPolar, referenceLogPolar_, logPolarMap_);
    auto transformed = getTransformed(img, transform);

    return {transformed, transform};
}

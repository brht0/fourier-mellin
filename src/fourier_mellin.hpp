#ifndef __FOURIER_MELLIN_H__
#define __FOURIER_MELLIN_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>

#include "fft.hpp"
#include "log_polar_map.hpp"
#include "filters.hpp"

struct Transform{
    double xOffset;
    double yOffset;
    double scale;
    double rotation;
    double response;
};

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform);

cv::Mat getProcessedImage(const cv::Mat &img, const cv::Mat& highPassFilter, const cv::Mat& apodizationWindow, const LogPolarMap& logPolarMap);

Transform registerGrayImage(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1, const LogPolarMap& logPolarMap);

class FourierMellin{
public:
    FourierMellin(int cols, int rows);
    ~FourierMellin();

    cv::Mat GetProcessImage(const cv::Mat &img) const;
    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1) const;
    // std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1);
    // Transform GetRegisteredImageTransform(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1, const LogPolarMap& logPolarMap);

private:
    int cols_, rows_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

};

#endif // __FOURIER_MELLIN_H__
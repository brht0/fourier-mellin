#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <opencv2/core/core.hpp>

struct LogPolarMap{
    int logPolarSize;
    double logBase;
    cv::Mat xMap;
    cv::Mat yMap;
};

LogPolarMap createLogPolarMap(int cols, int rows);

cv::Mat getLogPolarImage(const cv::Mat& img, const cv::Mat& polarMapX, const cv::Mat& polarMapY);

cv::Mat fft(const cv::Mat& img);

cv::Mat fftShift(const cv::Mat& in);

cv::Mat getHighPassFilter(int rows, int cols);

cv::Mat getApodizationWindow(int cols, int rows, int radius);

cv::Mat getFilteredImage(const cv::Mat &gray, const cv::Mat& apodizationWindow, const cv::Mat& highPassFilter);

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform);

cv::Mat getProcessedImage(const cv::Mat &img, const cv::Mat& highPassFilter, const cv::Mat& apodizationWindow, const LogPolarMap& logPolarMap);

Transform registerGrayImage(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &logPolar0, const cv::Mat &logPolar1, const LogPolarMap& logPolarMap);

#endif // __UTILITIES_H__
#ifndef __LOG_POLAR_MAP_H__
#define __LOG_POLAR_MAP_H__

#include <opencv2/opencv.hpp>

struct LogPolarMap{
    int logPolarSize;
    double logBase;
    cv::Mat xMap;
    cv::Mat yMap;
};

LogPolarMap createLogPolarMap(int cols, int rows);

cv::Mat getLogPolarImage(const cv::Mat& img, const cv::Mat& polarMapX, const cv::Mat& polarMapY);

#endif // __LOG_POLAR_MAP_H__
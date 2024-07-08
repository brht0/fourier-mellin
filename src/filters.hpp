#ifndef __FILTERS_H__
#define __FILTERS_H__

#include <opencv2/opencv.hpp>
#include "fft.hpp"

cv::Mat getHighPassFilter(int rows, int cols);

cv::Mat getApodizationWindow(int cols, int rows, int radius);

cv::Mat getFilteredImage(const cv::Mat &gray, const cv::Mat& apodizationWindow, const cv::Mat& highPassFilter);

#endif // __FILTERS_H__
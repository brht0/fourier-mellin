#ifndef __FFT_H__
#define __FFT_H__

#include <opencv2/core/core.hpp>

cv::Mat fft(const cv::Mat& img);
cv::Mat fftShift(const cv::Mat& in);

#endif // __FFT_H__
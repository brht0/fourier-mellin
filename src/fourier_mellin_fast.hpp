#ifndef __FOURIER_MELLIN_FAST_H__
#define __FOURIER_MELLIN_FAST_H__

#include <iostream>

#include "utilities.hpp"
#include "transform.hpp"

class FourierMellinFast{
public:
    FourierMellinFast(int cols, int rows);
    ~FourierMellinFast();

    void SetReference(const cv::Mat &img);
    Transform GetTransform(const cv::Mat &img) const;

private:
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

    cv::Mat reference_;
    cv::Mat referenceLogPolar_;
};

#endif // __FOURIER_MELLIN_FAST_H__
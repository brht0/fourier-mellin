#ifndef __FOURIER_MELLIN_H__
#define __FOURIER_MELLIN_H__

#include <iostream>

#include "utilities.hpp"
#include "transform.hpp"

class FourierMellin{
public:
    FourierMellin(int cols, int rows);
    ~FourierMellin();

    cv::Mat GetProcessImage(const cv::Mat &img) const;
    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img0, const cv::Mat &img1) const;

private:
    int cols_, rows_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;
};

class FourierMellinContinuous{
public:
    FourierMellinContinuous(int cols, int rows);
    ~FourierMellinContinuous();

    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img);

private:
    int cols_, rows_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

    bool isFirst_;
    cv::Mat prevGray_;
    cv::Mat prevLogPolar_;
    Transform transformSum_;
};

class FourierMellinWithReference{
public:
    FourierMellinWithReference(int cols, int rows);
    ~FourierMellinWithReference();

    void SetReference(const cv::Mat &img);
    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img);

private:
    int cols_, rows_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

    cv::Mat reference_;
    cv::Mat referenceLogPolar_;
};

#endif // __FOURIER_MELLIN_H__
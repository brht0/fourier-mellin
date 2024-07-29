#ifndef __FOURIER_MELLIN_H__
#define __FOURIER_MELLIN_H__

#include <iostream>
#include <map>

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
    FourierMellinContinuous(int cols, int rows, double edgeCrop = 0.1, double pullToCenterRatio = 0.07);
    ~FourierMellinContinuous();

    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img);

private:
    int cols_, rows_;
    double edgeCrop_;
    double pullToCenterRatio_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

    bool isFirst_;
    cv::Mat prevGray_;
    cv::Mat prevLogPolar_;
    Transform totalTransform_;
};

class FourierMellinWithReference{
public:
    FourierMellinWithReference(int cols, int rows);
    ~FourierMellinWithReference();

    void SetReference(const cv::Mat &img, int designation = -1);
    void SetReferenceWithDesignation(int designation);
    std::tuple<cv::Mat, Transform> GetRegisteredImage(const cv::Mat &img) const;
    Transform GetRegisteredImageTransform(const cv::Mat &img) const;

private:
    int cols_, rows_;

    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
    LogPolarMap logPolarMap_;

    int currentDesignation_;
    std::map<int, cv::Mat> references_;
    std::map<int, cv::Mat> referenceLogPolars_;
};

#endif // __FOURIER_MELLIN_H__
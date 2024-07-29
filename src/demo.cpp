#include "fourier_mellin.hpp"

#include <iomanip>

int main(){
    constexpr int cols = 1280;
    constexpr int rows = 720;
    
    auto highPassFilter = getHighPassFilter(rows, cols);
    auto apodizationWindow = getApodizationWindow(cols, rows, std::min(rows, cols));

    cv::Mat img0 = cv::imread("images/reference.jpg", cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread("images/transformed.jpg", cv::IMREAD_COLOR);
    
    // TODO: Resizing affects performance by huge amount, especially if non-Eucledian
    cv::resize(img0, img0, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_NEAREST);
    cv::resize(img1, img1, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_NEAREST);

    cv::Mat gray0, gray1;
    img0.convertTo(gray0, CV_32F, 1.0/255.0); cv::cvtColor(gray0, gray0, cv::COLOR_BGR2GRAY);
    img1.convertTo(gray1, CV_32F, 1.0/255.0); cv::cvtColor(gray1, gray1, cv::COLOR_BGR2GRAY);

    auto logPolarMap = createLogPolarMap(img0.cols, img0.rows);
    auto logPolar0 = getProcessedImage(gray0, highPassFilter, apodizationWindow, logPolarMap);
    auto logPolar1 = getProcessedImage(gray1, highPassFilter, apodizationWindow, logPolarMap);
    auto transform = registerGrayImage(gray0, gray1, logPolar0, logPolar1, logPolarMap);
    
    std::cout << transform << std::endl;

    cv::Mat registeredImage = getTransformed(img0, transform);
    cv::addWeighted(img1, 0.5, registeredImage, 0.5, 0.0, registeredImage);
    cv::imshow("Image Registration ", registeredImage);

    cv::waitKey(0);
    return 0;
}

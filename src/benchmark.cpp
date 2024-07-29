#include "fourier_mellin.hpp"

#include <iomanip>
#include <chrono>

int main(){
    constexpr int iterations = 7202;
    constexpr int scaleDownFactor = 16;
    constexpr int cols = 1280 / scaleDownFactor;
    constexpr int rows = 720 / scaleDownFactor;
    
    auto highPassFilter = getHighPassFilter(rows, cols);
    auto apodizationWindow = getApodizationWindow(cols, rows, std::min(rows, cols));

    cv::Mat img0 = cv::imread("images/reference2.jpg", cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread("images/transformed2.jpg", cv::IMREAD_COLOR);
    
    // cv::resize(img0, img0, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_NEAREST);
    // cv::resize(img1, img1, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_NEAREST);
    cv::resize(img0, img0, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_CUBIC);
    cv::resize(img1, img1, cv::Size(cols, rows), 0.0, 0.0, cv::InterpolationFlags::INTER_CUBIC);

    img0.convertTo(img0, CV_32F, 1.0/255.0);
    img1.convertTo(img1, CV_32F, 1.0/255.0);

    // cv::Mat gray0, gray1;
    // img0.convertTo(gray0, CV_32F, 1.0/255.0); cv::cvtColor(gray0, gray0, cv::COLOR_BGR2GRAY);
    // img1.convertTo(gray1, CV_32F, 1.0/255.0); cv::cvtColor(gray1, gray1, cv::COLOR_BGR2GRAY);

    FourierMellinWithReference fm(cols, rows);
    fm.SetReference(img0);
    auto startTime = std::chrono::high_resolution_clock::now();
    Transform transform;
    for(int i=0; i<iterations; i++){
        transform = fm.GetRegisteredImageTransform(img1);
        // std::cout << transform << "\n";
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto timeTakenSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1e-3;

    std::cout << "Time taken: " << timeTakenSeconds << ", transform: " << transform << "\n";

    return 0;
}

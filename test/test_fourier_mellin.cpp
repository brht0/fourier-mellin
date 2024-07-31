#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>

// TODO: Fix project include structure in src/CMakeLists.txt
#include "../src/fourier_mellin.hpp"
#include "../src/transform.hpp"

cv::Mat GetL2Difference(const cv::Mat& a, const cv::Mat& b){
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::Mat squaredDiff;
    cv::multiply(diff, diff, squaredDiff);
    return squaredDiff;
}

double DifferenceL2(const cv::Mat& a, const cv::Mat& b){
    if (a.size() != b.size() || a.type() != b.type()) {
        throw std::invalid_argument("Matrices must have the same size and type");
    }
    
    auto squaredDiff = GetL2Difference(a, b);

    double l2Diff = cv::sum(squaredDiff)[0];
    return l2Diff;
}

TEST(ImageTransformConsistency2, BasicAssertions) {
    Transform t_01(0, 0, 0.9, 15, 1);
    constexpr unsigned iterations = 10;
    constexpr bool saveFiles = false;

    auto img = cv::imread("images/lenna.png", cv::IMREAD_COLOR);
    EXPECT_NE(img.size(), cv::Size(0, 0));

    auto img_01 = img.clone();

    Transform t;
    for(int i=0; i<iterations; i++){
        t = t * t_01;

        img_01 = getTransformed(img_01, t_01);
        auto img_direct = getTransformed(img, t);

        double l2 = DifferenceL2(img_01, img_direct);

        if constexpr(saveFiles){
            std::ostringstream filename;
            filename << "output/img_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, img_01);
        }
        if constexpr(saveFiles){
            std::ostringstream filename;
            filename << "output/img_final_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, img_direct);
        }

        EXPECT_LE((l2 / (img.size().width * (double)img.size().height)), 15);
    }

    if constexpr(saveFiles){
        auto imgFinal = getTransformed(img, t);
        cv::imwrite("output/img_final.jpg", imgFinal);
    }
}

TEST(ImageTransformConsistency3, BasicAssertions) {
    Transform t_01(-10, 10, 0.9, 15, 1);
    constexpr unsigned iterations = 10;
    constexpr bool saveFiles = false;

    auto img = cv::imread("images/lenna.png", cv::IMREAD_COLOR);
    EXPECT_NE(img.size(), cv::Size(0, 0));

    auto img_01 = img.clone();

    Transform t;
    for(int i=0; i<iterations; i++){
        t = t * t_01;

        img_01 = getTransformed(img_01, t_01);
        auto img_direct = getTransformed(img, t);

        double l2 = DifferenceL2(img_01, img_direct);

        if constexpr(saveFiles){
            std::ostringstream filename;
            filename << "output/img_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, img_01);
        }
        if constexpr(saveFiles){
            std::ostringstream filename;
            filename << "output/img_final_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, img_direct);
        }
        if constexpr(saveFiles){
            auto imgDiff = GetL2Difference(img_01, img_direct);
            std::ostringstream filename;
            filename << "output/img_difference_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, imgDiff);
        }

        EXPECT_LE((l2 / (img.size().width * (double)img.size().height)), 15);
    }

    if constexpr(saveFiles){
        auto imgFinal = getTransformed(img, t);
        cv::imwrite("output/img_final.jpg", imgFinal);
    }
}

TEST(BasicFourierMellin1, BasicAssertions) {
    constexpr bool saveFiles = false;
    Transform t_01(-30, 20, 0.65, -20, 1);

    auto img = cv::imread("images/lenna_small_center.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_32FC(3));
    EXPECT_NE(img.size(), cv::Size(0, 0));

    auto img_01 = getTransformed(img, t_01);

    FourierMellin fm(img.size().width, img.size().height);
    auto[transformed,transform] = fm.GetRegisteredImage(img, img_01);

    double l2 = DifferenceL2(img_01, transformed);

    if constexpr(saveFiles){
        auto imgDiff = GetL2Difference(img_01, transformed);
        cv::imwrite("output/img_difference.jpg", imgDiff);
    }

    EXPECT_LE((l2 / (img.size().width * (double)img.size().height)), 15);

    EXPECT_NEAR(transform.GetScale(), t_01.GetScale(), 1e-2);
    EXPECT_NEAR(transform.GetOffsetX(), t_01.GetOffsetX(), 1e-0);
    EXPECT_NEAR(transform.GetOffsetY(), t_01.GetOffsetY(), 1e-0);
    EXPECT_NEAR(transform.GetRotation(), t_01.GetRotation(), 1e-0);
    EXPECT_GE(transform.GetResponse(), 0.5);
}

TEST(ChainedFourierMellin1, BasicAssertions) {
    constexpr bool saveFiles = true;
    constexpr unsigned iterations = 5;
    Transform t_01(-30, 20, 0.95, -15, 1);

    auto img = cv::imread("images/lenna_small_center.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_32FC(3));
    EXPECT_NE(img.size(), cv::Size(0, 0));

    FourierMellin fm(img.size().width, img.size().height);

    Transform t;
    for(unsigned i=0; i<iterations; i++){
        auto img_01 = getTransformed(img, t *= t_01);
        auto[transformed,transform] = fm.GetRegisteredImage(img, img_01);
        double l2 = DifferenceL2(img_01, transformed);

        EXPECT_LE((l2 / (img.size().width * (double)img.size().height)), 15);
        EXPECT_NEAR(transform.GetScale(), t.GetScale(), 1e-2);
        EXPECT_NEAR(transform.GetOffsetX(), t.GetOffsetX(), 1e-0);
        EXPECT_NEAR(transform.GetOffsetY(), t.GetOffsetY(), 1e-0);
        EXPECT_NEAR(transform.GetRotation(), t.GetRotation(), 1e-0);
        EXPECT_GE(transform.GetResponse(), 0.5);

        if constexpr(saveFiles){
            auto imgDiff = GetL2Difference(img_01, transformed);
            std::ostringstream filename;
            filename << "output/img_difference_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, imgDiff);
        }
    }
}

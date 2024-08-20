#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>

// TODO: Fix project include structure in src/CMakeLists.txt
#include "../src/fourier_mellin.hpp"
#include "../src/transform.hpp"

cv::Mat GetL2Difference(const cv::Mat& a, const cv::Mat& b){
    cv::Mat mask = (a == 0) | (b == 0);
    mask.convertTo(mask, CV_32F);
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    cv::Mat squaredDiff;
    cv::multiply(diff, diff, squaredDiff);
    return squaredDiff.mul(1.f - mask);
}

cv::Mat GetAverageImage(const cv::Mat& a, const cv::Mat& b){
    return a * 0.5 + b * 0.5;
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
    img.convertTo(img, CV_32F);
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
    img.convertTo(img, CV_32F);
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
    constexpr bool saveFiles = false;
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

TEST(ChainedFourierMellinRandom1, BasicAssertions) {
    constexpr bool saveFiles = false;
    constexpr unsigned iterations = 5;

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> scale(0.8, 1.1);
    std::uniform_real_distribution<> rotation(-25, 25);
    std::uniform_real_distribution<> xOffset(-50, 50);
    std::uniform_real_distribution<> yOffset(-50, 50);

    auto img = cv::imread("images/lenna_small_center.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_32FC(3));
    EXPECT_NE(img.size(), cv::Size(0, 0));

    FourierMellin fm(img.size().width, img.size().height);

    Transform t;
    for(unsigned i=0; i<iterations; i++){
        Transform t1(xOffset(e2), yOffset(e2), scale(e2), rotation(e2), 1.0);
        auto img_01 = getTransformed(img, t *= t1);
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

TEST(ChainedFourierMellin2, BasicAssertions) {
    constexpr bool saveFiles = false;
    constexpr unsigned iterations = 10;

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> scaleCumulative(1/1.1, 1.1);
    std::uniform_real_distribution<> rotationCumulative(-5, 5);
    std::uniform_real_distribution<> xOffsetCumulative(-25, 25);
    std::uniform_real_distribution<> yOffsetCumulative(-25, 25);

    std::uniform_real_distribution<> scale(0.8, 1.2);
    std::uniform_real_distribution<> rotation(-45, 45);
    std::uniform_real_distribution<> xOffset(-75, 75);
    std::uniform_real_distribution<> yOffset(-75, 75);

    auto img = cv::imread("images/lenna_small_center.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_32FC(3));
    EXPECT_NE(img.size(), cv::Size(0, 0));

    FourierMellin fm(img.size().width, img.size().height);

    Transform t;
    for(unsigned i=0; i<iterations; i++){
        Transform tImmediate(xOffsetCumulative(e2), yOffsetCumulative(e2), scaleCumulative(e2), rotationCumulative(e2), 1.0);
        Transform tTest(xOffset(e2), yOffset(e2), scale(e2), rotation(e2), 1.0);
        t *= tImmediate;

        auto img_01 = getTransformed(img, t);
        auto[transformed,transform] = fm.GetRegisteredImage(img, img_01);
        double l2 = DifferenceL2(img_01, transformed);

        auto imgTest = getTransformed(img_01, tTest);
        auto[transformed2,transform2] = fm.GetRegisteredImage(img_01, imgTest);

        EXPECT_LE((l2 / (img.size().width * (double)img.size().height)), 15);
        EXPECT_NEAR(transform.GetScale(), t.GetScale(), 1e-2);
        EXPECT_NEAR(transform.GetOffsetX(), t.GetOffsetX(), 1e-0);
        EXPECT_NEAR(transform.GetOffsetY(), t.GetOffsetY(), 1e-0);
        EXPECT_NEAR(transform.GetRotation(), t.GetRotation(), 1e-0);
        EXPECT_GE(transform.GetResponse(), 0.5);

        EXPECT_NEAR(transform2.GetScale(), tTest.GetScale(), 1e-2);
        EXPECT_NEAR(transform2.GetOffsetX(), tTest.GetOffsetX(), 1e-0);
        EXPECT_NEAR(transform2.GetOffsetY(), tTest.GetOffsetY(), 1e-0);
        EXPECT_NEAR(transform2.GetRotation(), tTest.GetRotation(), 1e-0);
        EXPECT_GE(transform2.GetResponse(), 0.5);

        if constexpr(saveFiles){
            auto imgDiff = GetL2Difference(img_01, transformed);
            std::ostringstream filename;
            filename << "output/img_difference_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, imgDiff);
        }
    }
}

TEST(ChainedFourierMellin_DirectAndCumulative1, BasicAssertions) {
    constexpr bool saveFiles = false;
    constexpr unsigned iterations = 10;

    std::vector<cv::Mat> imgs;

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> scale(0.5, 1.1);
    std::uniform_real_distribution<> rotation(-25, 25);
    std::uniform_real_distribution<> xOffset(-50, 50);
    std::uniform_real_distribution<> yOffset(-50, 50);

    // auto img = cv::imread("images/lenna_small_center.png", cv::IMREAD_COLOR);
    auto img = cv::imread("images/lenna.png", cv::IMREAD_COLOR);
    img.convertTo(img, CV_32FC(3));
    EXPECT_NE(img.size(), cv::Size(0, 0));

    FourierMellin fm(img.size().width, img.size().height);

    for(unsigned i=0; i<iterations; i++){
        Transform t(xOffset(e2), yOffset(e2), scale(e2), rotation(e2), 1.0);
        auto imgTransformed = getTransformed(img, t);
        imgs.push_back(imgTransformed);
    }

    Transform t;
    for(unsigned i=0; i<iterations; i++){
        const auto& imgBase = (i == 0 ? img : imgs[i-1]);
        const auto& imgTarget = imgs[i];

        auto[transformedImmediate,tImmediate] = fm.GetRegisteredImage(imgBase, imgTarget);
        auto[transformedDirect,transformDirect] = fm.GetRegisteredImage(img, imgTarget);

        t = tImmediate * t;

        EXPECT_NEAR(transformDirect.GetScale(), t.GetScale(), 2.0e-2);
        EXPECT_NEAR(transformDirect.GetOffsetX(), t.GetOffsetX(), 2.0e-0);
        EXPECT_NEAR(transformDirect.GetOffsetY(), t.GetOffsetY(), 2.0e-0);
        EXPECT_NEAR(transformDirect.GetRotation(), t.GetRotation(), 2.0e-0);
        EXPECT_GE(transformDirect.GetResponse(), 0.5);
        EXPECT_GE(tImmediate.GetResponse(), 0.5);

        if constexpr(saveFiles){
            auto imgDiff = GetL2Difference(transformedImmediate, transformedDirect);
            std::ostringstream filename;
            filename << "output/img_difference_" << std::setw(2) << std::setfill('0') << i << ".jpg";
            std::string filenameStr = filename.str();
            cv::imwrite(filenameStr, imgDiff);
        }
    }
}

TEST(FourierMellin_ImageFeed1, BasicAssertions) {
    auto readImage = [](const std::string& fn){
        cv::Mat img = cv::imread(fn, cv::IMREAD_COLOR);
        img.convertTo(img, CV_32FC(3));
        EXPECT_NE(img.size(), cv::Size(0, 0));
        return img;
    };

    auto expectTransformsNear = [](const std::vector<Transform>& ts){
        for(size_t i=0; i<ts.size(); i++){
            for(size_t j=0; j<ts.size(); j++){
                if(i == j)
                    continue;
                EXPECT_NEAR(ts[i].GetScale(), ts[j].GetScale(), 2.0e-2);
                EXPECT_NEAR(ts[i].GetOffsetX(), ts[j].GetOffsetX(), 2.0e-0);
                EXPECT_NEAR(ts[i].GetOffsetY(), ts[j].GetOffsetY(), 2.0e-0);
                EXPECT_NEAR(ts[i].GetRotation(), ts[j].GetRotation(), 2.0e-0);
            }
        }
    };

    auto expectImagesNear = [](const cv::Mat& img1, const cv::Mat& img2){
        static int i=0;
        double l2 = DifferenceL2(img1, img2) / (img1.size().width * img1.size().height);
        EXPECT_LE(l2, 20);

        auto imgDiff = GetL2Difference(img1, img2);
        std::ostringstream filename;
        filename << "output/img_difference_" << (i++) << ".jpg";
        std::string filenameStr = filename.str();
        cv::imwrite(filenameStr, imgDiff);
    };

    namespace fs = std::filesystem;

    std::string fn0 = "images/image_feed/frame_0020.jpg";
    std::string fn1 = "images/image_feed/frame_0049.jpg";
    // std::string fn1 = "images/image_feed/frame_0203.jpg";
    std::string fn2 = "images/image_feed/frame_0386.jpg";
    // std::string fn3 = "images/image_feed/frame_0505.jpg";
    std::string fn3 = "images/image_feed/frame_0564.jpg";

    auto img0 = readImage(fn0);
    auto img1 = readImage(fn1);
    auto img2 = readImage(fn2);
    auto img3 = readImage(fn3);

    FourierMellin fm(img0.size().width, img0.size().height);

    auto[img_00, t_00] = fm.GetRegisteredImage(img0, img0);
    auto[img_01, t_01] = fm.GetRegisteredImage(img0, img1);
    auto[img_02, t_02] = fm.GetRegisteredImage(img0, img2);
    auto[img_03, t_03] = fm.GetRegisteredImage(img0, img3);
    
    auto[img_10, t_10] = fm.GetRegisteredImage(img1, img0);
    auto[img_11, t_11] = fm.GetRegisteredImage(img1, img1);
    auto[img_12, t_12] = fm.GetRegisteredImage(img1, img2);
    auto[img_13, t_13] = fm.GetRegisteredImage(img1, img3);
    
    auto[img_20, t_20] = fm.GetRegisteredImage(img2, img0);
    auto[img_21, t_21] = fm.GetRegisteredImage(img2, img1);
    auto[img_22, t_22] = fm.GetRegisteredImage(img2, img2);
    auto[img_23, t_23] = fm.GetRegisteredImage(img2, img3);
    
    auto[img_30, t_30] = fm.GetRegisteredImage(img3, img0);
    auto[img_31, t_31] = fm.GetRegisteredImage(img3, img1);
    auto[img_32, t_32] = fm.GetRegisteredImage(img3, img2);
    auto[img_33, t_33] = fm.GetRegisteredImage(img3, img3);

    EXPECT_GE(t_00.GetResponse(), 0.99);
    EXPECT_GE(t_11.GetResponse(), 0.99);
    EXPECT_GE(t_22.GetResponse(), 0.99);
    EXPECT_GE(t_33.GetResponse(), 0.99);
    EXPECT_GE(t_00.GetResponse(), 0.25);
    EXPECT_GE(t_01.GetResponse(), 0.25);
    EXPECT_GE(t_02.GetResponse(), 0.25);
    EXPECT_GE(t_03.GetResponse(), 0.25);
    EXPECT_GE(t_10.GetResponse(), 0.25);
    EXPECT_GE(t_11.GetResponse(), 0.25);
    EXPECT_GE(t_12.GetResponse(), 0.25);
    EXPECT_GE(t_13.GetResponse(), 0.25);
    EXPECT_GE(t_20.GetResponse(), 0.25);
    EXPECT_GE(t_21.GetResponse(), 0.25);
    EXPECT_GE(t_22.GetResponse(), 0.25);
    EXPECT_GE(t_23.GetResponse(), 0.25);
    EXPECT_GE(t_30.GetResponse(), 0.25);
    EXPECT_GE(t_31.GetResponse(), 0.25);
    EXPECT_GE(t_32.GetResponse(), 0.25);
    EXPECT_GE(t_33.GetResponse(), 0.25);

    expectTransformsNear({t_00, t_11, t_22, t_33});
    expectTransformsNear({t_01, t_10.GetInverse()});
    expectTransformsNear({t_02, t_20.GetInverse()});
    expectTransformsNear({t_03, t_30.GetInverse()});

    expectTransformsNear({t_10, t_01.GetInverse()});
    expectTransformsNear({t_12, t_21.GetInverse()});
    expectTransformsNear({t_13, t_31.GetInverse()});

    expectTransformsNear({t_20, t_02.GetInverse()});
    expectTransformsNear({t_21, t_12.GetInverse()});
    expectTransformsNear({t_23, t_32.GetInverse()});

    expectTransformsNear({t_30, t_03.GetInverse()});
    expectTransformsNear({t_31, t_13.GetInverse()});
    expectTransformsNear({t_32, t_23.GetInverse()});

    expectImagesNear(img_00, img0);
    expectImagesNear(img_11, img1);
    expectImagesNear(img_22, img2);
    expectImagesNear(img_33, img3);

    expectImagesNear(getTransformed(img0, t_01), img1);
    expectImagesNear(getTransformed(img0, t_02), img2);
    expectImagesNear(getTransformed(img0, t_03), img3);
    
    expectImagesNear(getTransformed(img1, t_10), img0);
    expectImagesNear(getTransformed(img1, t_12), img2);
    expectImagesNear(getTransformed(img1, t_13), img3);
    
    expectImagesNear(getTransformed(img2, t_20), img0);
    expectImagesNear(getTransformed(img2, t_21), img1);
    expectImagesNear(getTransformed(img2, t_23), img3);
    
    expectImagesNear(getTransformed(img3, t_30), img0);
    expectImagesNear(getTransformed(img3, t_31), img1);
    expectImagesNear(getTransformed(img3, t_32), img2);

    expectImagesNear(getTransformed(img_01, t_10), img0);
    expectImagesNear(getTransformed(img_02, t_20), img0);
    expectImagesNear(getTransformed(img_03, t_30), img0);

    expectImagesNear(getTransformed(img_10, t_01), img1);
    expectImagesNear(getTransformed(img_12, t_21), img1);
    expectImagesNear(getTransformed(img_13, t_31), img1);

    expectImagesNear(getTransformed(img_20, t_02), img2);
    expectImagesNear(getTransformed(img_21, t_12), img2);
    expectImagesNear(getTransformed(img_23, t_32), img2);

    expectImagesNear(getTransformed(img_30, t_03), img3);
    expectImagesNear(getTransformed(img_31, t_13), img3);
    expectImagesNear(getTransformed(img_32, t_23), img3);

    auto img_01_12_23 = getTransformed(getTransformed(getTransformed(img0, t_01), t_12), t_23);
    auto img_02_21_13 = getTransformed(getTransformed(getTransformed(img0, t_02), t_21), t_13);
    expectImagesNear(img_01_12_23, img_02_21_13);

    auto img_03_31_12_23_30 = getTransformed(getTransformed(getTransformed(getTransformed(getTransformed(img0, t_03), t_31), t_12), t_23), t_30);
    expectImagesNear(img_03_31_12_23_30, img0);
}

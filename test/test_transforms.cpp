#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>

// TODO: Fix project include structure in src/CMakeLists.txt
#include "../src/transform.hpp"

TEST(DefaultConstructor, BasicAssertions) {
    Transform t;
    EXPECT_DOUBLE_EQ(t.GetOffsetX(), 0.0);
    EXPECT_DOUBLE_EQ(t.GetOffsetY(), 0.0);
    EXPECT_DOUBLE_EQ(t.GetScale(), 1.0);
    EXPECT_DOUBLE_EQ(t.GetRotation(), 0.0);
    EXPECT_DOUBLE_EQ(t.GetResponse(), 0.0);
}

TEST(Rotation, BasicAssertions) {
    constexpr double r1 = 17;
    constexpr double r2 = 22;
    Transform t1(0, 0, 1, r1, 0);
    Transform t2(0, 0, 1, r2, 0);

    constexpr double r3 = 2*r1+3*r2;
    auto t3 = t1 * t2 * t2 * t1 * t2;

    Transform t4(0, 0, 1, r3, 0);

    EXPECT_NEAR(t3.GetOffsetX(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t4.GetRotation(), r3, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), r3, 1e-6);
}

TEST(ScaleAndRotation, BasicAssertions) {
    constexpr double r1 = 17;
    constexpr double r2 = 22;
    constexpr double s1 = 1.1;
    constexpr double s2 = 0.93;
    Transform t1(0, 0, s1, r1, 0);
    Transform t2(0, 0, s2, r2, 0);

    constexpr double r3 = 2*r1+3*r2;
    constexpr double s3 = s1*s1*s2*s2*s2;
    auto t3 = t1 * t2 * t2 * t1 * t2;

    Transform t4(0, 0, s3, r3, 0);

    EXPECT_NEAR(t3.GetOffsetX(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetScale(), s3, 1e-6);
    EXPECT_NEAR(t4.GetScale(), s3, 1e-6);
    EXPECT_NEAR(t4.GetRotation(), r3, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), r3, 1e-6);
}

TEST(Offset, BasicAssertions) {
    constexpr double x1 = 17;
    constexpr double x2 = 22;
    constexpr double y1 = 12;
    constexpr double y2 = 4;
    Transform t1(x1, y1, 1.0, 0.0, 0.0);
    Transform t2(x2, y2, 1.0, 0.0, 0.0);

    constexpr double x3 = 2*x1+3*x2;
    constexpr double y3 = 2*y1+3*y2;
    auto t3 = t1 * t2 * t2 * t1 * t2;

    Transform t4(x3, y3, 1.0, 0.0, 0.0);

    EXPECT_NEAR(t3.GetOffsetX(), x3, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), y3, 1e-6);
    EXPECT_NEAR(t4.GetOffsetX(), x3, 1e-6);
    EXPECT_NEAR(t4.GetOffsetY(), y3, 1e-6);
    EXPECT_NEAR(t3.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t4.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t4.GetRotation(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), 0.0, 1e-6);
}

TEST(RotationAndOffsets1, BasicAssertions) {
    constexpr double x = 3;
    constexpr double r = 90;
    Transform t1(0.0, 0.0, 1.0, 0.0, 0.0);
    Transform t2(x, 0.0, 1.0, r, 0.0);

    double x2 = 0;
    double y2 = x;
    double r2 = r*3 - 360.0;

    auto t3 = t1 * t2 * t2 * t2;

    EXPECT_NEAR(t3.GetOffsetX(), x2, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), y2, 1e-6);
    EXPECT_NEAR(t3.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), r2, 1e-6);
}

TEST(FullRotation_1, BasicAssertions) {
    constexpr double x = 1.2345;
    constexpr double r = 45;
    Transform t1(0.0, 0.0, 1.0, 0.0, 0.0);
    Transform t2(x, 0.0, 1.0, r, 0.0);

    auto t3 = t1;
    for(int i=0; i<8; i++){
        t3 *= t2;
    }

    EXPECT_NEAR(t3.GetOffsetX(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), 0.0, 1e-6);
}

TEST(FullRotation_2, BasicAssertions) {
    constexpr double x = 100;
    constexpr double y = -30;
    constexpr double r = 360.0/100.0;
    Transform t1(0.0, 0.0, 1.0, 0.0, 0.0);
    Transform t2(x, y, 1.0, r, 0.0);

    auto t3 = t1;
    for(int i=0; i<100; i++){
        t3 *= t2;
    }

    EXPECT_NEAR(t3.GetOffsetX(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), 0.0, 1e-6);
    EXPECT_NEAR(t3.GetScale(), 1.0, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), 0.0, 1e-6);
}

TEST(Inverses, BasicAssertions) {
    constexpr double x = -12;
    constexpr double y = 34;
    constexpr double s = 1.234;
    constexpr double r = -12.34;
    Transform t1(0.0, 0.0, 1.0, 0.0, 0.0);
    Transform t2(x, y, s, r, 0.0);
    auto m2 = t2.GetMatrix();
    auto m2Inv = t2.GetMatrixInverse();

    auto m3 = t1.GetMatrix();
    for(int i=0; i<2; i++){
        m3 *= m2;
    }
    for(int i=0; i<4; i++){
        m3 *= m2Inv;
    }
    for(int i=0; i<2; i++){
        m3 *= m2;
    }

    Transform t3(m3, 0.0);

    EXPECT_NEAR(t3.GetOffsetX(), t1.GetOffsetX(), 1e-6);
    EXPECT_NEAR(t3.GetOffsetY(), t1.GetOffsetY(), 1e-6);
    EXPECT_NEAR(t3.GetScale(), t1.GetScale(), 1e-6);
    EXPECT_NEAR(t3.GetRotation(), t1.GetRotation(), 1e-6);
}

TEST(RandomWalk, BasicAssertions) {
    std::vector<double> xs = {3.14, 2.71, 1.41, 0.57, 4.67, 5.23, 6.78, 7.89, 8.12, 9.34};
    std::vector<double> ys = {0.45, 1.67, 2.89, 3.01, 4.23, 5.34, 6.45, 7.56, 8.67, 9.78};
    std::vector<double> rs = {0.12, 1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01};
    std::vector<double> ss = {1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77, 8.88, 9.99, 10.10};

    Transform t3;
    for(int i=0; i<10; i++){
        t3 *= Transform(xs[i], ys[i], ss[i], rs[i], 0.0);
    }

    double r3 = std::accumulate(rs.begin(), rs.end(), 0.0);
    double s3 = std::accumulate(ss.begin(), ss.end(), 1.0, [](double a, double b){ return a * b; });

    EXPECT_NEAR(t3.GetScale(), s3, 1e-6);
    EXPECT_NEAR(t3.GetRotation(), r3, 1e-6);
}

#include "fft.hpp"

cv::Mat fft(const cv::Mat& img) {
    cv::Mat planes[] = {cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);
    return complex;
}

cv::Mat fftShift(const cv::Mat& in) {
    cv::Mat out = in.clone();
    int cx = in.cols / 2;
    int cy = in.rows / 2;

    int cx1 = (in.cols % 2 == 0) ? cx : cx + 1;
    int cy1 = (in.rows % 2 == 0) ? cy : cy + 1;

    cv::Mat q0(out, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(out, cv::Rect(cx, 0, cx1, cy));
    cv::Mat q2(out, cv::Rect(0, cy, cx, cy1));
    cv::Mat q3(out, cv::Rect(cx, cy, cx1, cy1));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return out;
}

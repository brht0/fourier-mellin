#include "transform.hpp"
#include <numbers>
#include <iomanip>

Transform::Transform(double xOffset, double yOffset, double scale, double rotationDeg, double response):
    xOffset_(xOffset),
    yOffset_(yOffset),
    scale_(scale),
    rotation_(rotationDeg),
    response_(response)
{
}

Transform::Transform(const cv::Mat& matrix, double response){
    CV_Assert(matrix.rows == 3 && matrix.cols == 3);

    xOffset_ = matrix.at<double>(0, 2);
    yOffset_ = matrix.at<double>(1, 2);

    scale_ = std::hypot(matrix.at<double>(0, 0), matrix.at<double>(0, 1));
    rotation_ = std::atan2(matrix.at<double>(1, 0), matrix.at<double>(0, 0)) * (180.0/std::numbers::pi_v<double>);
    response_ = response;
}

cv::Mat Transform::GetMatrix() const {
    cv::Mat transform = cv::Mat::zeros(3, 3, CV_64F);

    double c = std::cos(rotation_ * (std::numbers::pi_v<double>/180.0));
    double s = std::sin(rotation_ * (std::numbers::pi_v<double>/180.0));

    transform.at<double>(0, 0) = scale_ * c;
    transform.at<double>(0, 1) = -scale_ * s;
    transform.at<double>(0, 2) = xOffset_;
    transform.at<double>(1, 0) = scale_ * s;
    transform.at<double>(1, 1) = scale_ * c;
    transform.at<double>(1, 2) = yOffset_;
    transform.at<double>(2, 0) = 0.0;
    transform.at<double>(2, 1) = 0.0;
    transform.at<double>(2, 2) = 1.0;

    return transform;
}

cv::Mat Transform::GetMatrixInverse() const {
    return GetMatrix().inv();
}

double Transform::GetOffsetX() const {
    return xOffset_;
}

double Transform::GetOffsetY() const {
    return yOffset_;
}

double Transform::GetScale() const {
    return scale_;
}

double Transform::GetRotation() const {
    return rotation_;
}

double Transform::GetResponse() const {
    return response_;
}

Transform Transform::operator*(const Transform& rhs) const {
    // double response = (response_ + rhs.response_) * 0.5;
    double response = std::min(response_, rhs.response_);
    return Transform(GetMatrix() * rhs.GetMatrix(), response);
}

std::ostream& operator<<(std::ostream& os, const Transform& t){
    os << std::fixed << std::setprecision(2) << "Transform(" << t.GetOffsetX() << ", " << t.GetOffsetY() << ", " << t.GetScale() << ", " << t.GetRotation() << ", " << t.GetResponse() << ")";
    return os;
}

void Transform::SetOffsetX(double x) {
    xOffset_ = x;
}

void Transform::SetOffsetY(double y) {
    yOffset_ = y;
}

void Transform::SetScale(double scale) {
    scale_ = scale;
}

void Transform::SetRotation(double rotationDeg) {
    rotation_ = rotationDeg;
}

void Transform::SetResponse(double response) {
    response_ = response;
}

Transform& Transform::operator*=(const Transform& rhs){
    *this = *this * rhs;
    return *this;
}

Transform Transform::GetInverse() const {
    return Transform(GetMatrixInverse(), response_);
}

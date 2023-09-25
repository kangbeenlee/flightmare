#include "flightlib/tracking_algorithm/kalman_filter.hpp"

namespace flightlib {

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::reset()
{
    x_ = (Vector<9>() << 0, 0, 0, 0, 0, 0, 0, 0, 0).finished(); // w.r.t. camera frame
    P_ = Matrix<9, 9>::Identity() * 10.0;
    initialized_ = false;
}

void KalmanFilter::init(const Scalar Ts, Ref<Vector<9>> x0, const Scalar sigma_w, const Scalar sigma_v)
{
    // Kalman filter sampling time
    Ts_ = Ts;

    // Initial guess
    x_ = x0;
    P_ = Matrix<9, 9>::Identity() * 10.0;

    // System matrix for constant acceleration model
    F_.setZero();
    Matrix<3, 3> F_base = (Matrix<3, 3>() << 1, Ts_, 0.5 * pow(Ts_, 2),
                                             0,   1,               Ts_,
                                             0,   0,                 1).finished();
    F_.block<3,3>(0,0) = F_base;
    F_.block<3,3>(3,3) = F_base;
    F_.block<3,3>(6,6) = F_base;

    // Gamma.shape = (9, 3)
    Gamma_.setZero();
    Matrix<3, 1> Gamma_base = (Matrix<3, 1>() << pow(Ts_, 3) / 6, pow(Ts_, 2) / 2, Ts_).finished();
    Gamma_.block<3, 1>(0,0) = Gamma_base;
    Gamma_.block<3, 1>(3,1) = Gamma_base;
    Gamma_.block<3, 1>(6,2) = Gamma_base;

    // Measurement matrix
    H_.setZero();
    H_(0, 0) = 1;
    H_(1, 3) = 1;
    H_(2, 6) = 1;


    // Define system noise matrix
    Q_ = Matrix<3, 3>::Identity() * pow(sigma_w, 2);
    // Define sensor noise matrix
    R_ = Matrix<3, 3>::Identity() * pow(sigma_v, 2);

    initialized_ = true;
}

void KalmanFilter::predict()
{
    if (!initialized_) throw std::runtime_error("Filter is not initialized!");

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Gamma_ * Q_ * Gamma_.transpose();
}

void KalmanFilter::update(const Ref<Vector<3>> z)
{
    if (!initialized_) throw std::runtime_error("Filter is not initialized!");

    // Innovation
    Matrix<3, 3> S = H_ * P_ * H_.transpose() + R_;
    // Kalman gain
    K_ = P_ * H_.transpose() * S.inverse();
    // State update
    x_ = x_ + K_ * (z - H_ * x_);
    // Covariance update
    P_ = (I_ - K_ * H_) * P_;
}

Vector<3> KalmanFilter::computeEstimatedPositionWrtWorld(Ref<Matrix<4, 4>> T_LC_W)
{
    // To homogeneous coordinates
    Vector<4> P_C(x_[0], x_[3], x_[6], 1);

    // Target coordinates w.r.t. body frame
    Vector<3> p_w = (T_LC_W * P_C).segment<3>(0);

    return p_w;
}

Vector<3> KalmanFilter::computeEstimatedVelocityWrtWorld(Ref<Matrix<4, 4>> T_LC_W)
{
    // To homogeneous coordinates
    Vector<4> V_C(x_[1], x_[4], x_[7], 1);

    // Target coordinates w.r.t. body frame
    Vector<3> v_w = (T_LC_W * V_C).segment<3>(0);

    return v_w;
}

Scalar KalmanFilter::computeRangeWrtBody(Ref<Vector<3>> from, Ref<Matrix<4, 4>> T_LC_B)
{
    // To homogeneous coordinates
    Vector<4> P_C(x_[0], x_[3], x_[6], 1);

    // Target coordinates w.r.t. body frame
    Vector<3> p_b = (T_LC_B * P_C).segment<3>(0);

    Scalar range = sqrt(pow(p_b[0] - from[0], 2) + pow(p_b[1] - from[1], 2) + pow(p_b[2] - from[2], 2));

    return range;
}

}  // namespace flightlib
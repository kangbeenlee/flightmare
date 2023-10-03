#include "flightlib/tracking_algorithm/kalman_filter.hpp"

namespace flightlib {

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::reset()
{
    x_ = (Vector<6>() << 0, 0, 0, 0, 0, 0).finished(); // w.r.t. camera frame
    P_ = Matrix<6, 6>::Identity();
}

void KalmanFilter::init(const Scalar Ts, Ref<Vector<6>> x0)
{
    // Kalman filter sampling time
    Ts_ = Ts;

    // Initial guess
    x_ = x0;
    P_ = Matrix<6, 6>::Identity();

    // System matrix for constant acceleration model
    F_.setZero();
    Matrix<2, 2> F_base = (Matrix<2, 2>() << 1, Ts_, 
                                             0,   1).finished();
    F_.block<2, 2>(0,0) = F_base;
    F_.block<2, 2>(2,2) = F_base;
    F_.block<2, 2>(4,4) = F_base;

    // Gamma.shape = (6, 3)
    Gamma_.setZero();
    Matrix<2, 1> Gamma_base = (Matrix<2, 1>() << pow(Ts_, 2) / 2, Ts_).finished();
    Gamma_.block<2, 1>(0,0) = Gamma_base;
    Gamma_.block<2, 1>(2,1) = Gamma_base;
    Gamma_.block<2, 1>(4,2) = Gamma_base;

    // Measurement matrix
    H_.setZero();
    H_(0, 0) = 1;
    H_(1, 2) = 1;
    H_(2, 4) = 1;

    // Define system noise matrix
    Q_ = Matrix<3, 3>::Identity() * pow(sigma_w_, 2);
    // Define sensor noise matrix
    R_ = Matrix<3, 3>::Identity() * pow(sigma_v_, 2);

    initialized_ = true;
}

void KalmanFilter::predict()
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Gamma_ * Q_ * Gamma_.transpose();
}

void KalmanFilter::update(const Ref<Vector<3>> z, const Ref<Vector<3>> ego)
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    // Adaptive sensor noise
    if (adaptive_) {
        Vector<3> rel = Vector<3>(abs(z[0] - ego[0]), abs(z[1] - ego[1]), abs(z[2] - ego[2]));
        std::cout << "Before scaled std : " << rel[0] << ", " << rel[1] << ", " << rel[2] << std::endl;

        Scalar scale = 5.0;
        Vector<3> sigma_v = rel * scale;
        R_ = rel.asDiagonal();

        std::cout << "Adaptive sensor std : " << sigma_v[0] << ", " << sigma_v[1] << ", " << sigma_v[2] << std::endl;
    }

    // Innovationf
    Matrix<3, 3> S = H_ * P_ * H_.transpose() + R_;
    // Kalman gain
    K_ = P_ * H_.transpose() * S.inverse();
    // State update
    x_ = x_ + K_ * (z - H_ * x_);
    // Covariance update
    P_ = (I_ - K_ * H_) * P_;
}

}  // namespace flightlib
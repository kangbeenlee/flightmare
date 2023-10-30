#include "flightlib/tracking_algorithm/kalman_filter.hpp"

namespace flightlib {

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::reset()
{
    x_ = (Vector<6>() << 0, 0, 0, 0, 0, 0).finished(); // w.r.t. camera frame
    P_ = Matrix<6, 6>::Identity() * 5.0;
}

void KalmanFilter::init(const Scalar Ts, Ref<Vector<6>> x0)
{
    // Kalman filter sampling time
    Ts_ = Ts;

    // Initial guess
    x_ = x0;
    P_ = Matrix<6, 6>::Identity() * 5.0;

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

    initialized_ = true;
}

void KalmanFilter::predict()
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Gamma_ * Q_ * Gamma_.transpose();
}

void KalmanFilter::update(const Ref<Vector<3>> z)
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    // Error covariance induced by sensor measurement in 3D space is computed
    // by propagating the measurement error covariance through the measurement model

    Vector<3> unit_z = z / (z.norm() + 1e-8);

    // Scalar x_c = unit_z(0);
    // Scalar y_c = unit_z(1);
    // Scalar z_c = unit_z(2);

    Scalar x_c = z(0);
    Scalar y_c = z(1);
    Scalar z_c = z(2);
    
    // Jacobian matrix of the measurement model with respect to the 3D position
    Matrix<3, 3> J = (Matrix<3, 3>() << z_c,    0,  -x_c*z_c,
                                          0,  z_c,  -y_c*z_c,
                                          0,    0,  -z_c*z_c).finished(); // w.r.t. camera frame

    Vector<3> square_sigma_v(0.4, 0.4, 1.0); // squared form
    Matrix<3, 3> D = square_sigma_v.asDiagonal();
    Matrix<3, 3> R = J * D * J.transpose();

    // Innovationf
    Matrix<3, 3> S = H_ * P_ * H_.transpose() + R;
    // Kalman gain
    K_ = P_ * H_.transpose() * S.inverse();
    // State update
    x_ = x_ + K_ * (z - H_ * x_);
    // Covariance update
    P_ = (I_ - K_ * H_) * P_;
}

Matrix<3, 3> KalmanFilter::getPositionErrorCovariance() const
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");
    Matrix<3, 3> state_cov = (Matrix<3, 3>() << P_(0, 0), P_(0, 2), P_(0, 4),
                                                P_(2, 0), P_(2, 2), P_(2, 4),
                                                P_(4, 0), P_(4, 2), P_(4, 4)).finished(); // w.r.t. camera frame
    return state_cov;
}

}  // namespace flightlib
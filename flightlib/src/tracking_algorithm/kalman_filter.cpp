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
    sigma_w_ = 10.0;
    Q_ = Matrix<3, 3>::Identity() * pow(sigma_w_, 2);

    // sigma_alpha_ = 0.3 / 733; // sigma_alpha = sigma_u / focal_length
    // sigma_beta_ = 0.3 / 733; // sigma_beta = sigma_v / focal_length
    sigma_alpha_ = 0.05;
    sigma_beta_ = 0.05;
    sigma_rho_ = 0.03;

    Vector<3> square_sigma_v(pow(sigma_alpha_, 2), pow(sigma_beta_, 2), pow(sigma_rho_, 2)); // squared form
    D_ = square_sigma_v.asDiagonal();


    initialized_ = true;
}

void KalmanFilter::predict()
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Gamma_ * Q_ * Gamma_.transpose();
}

void KalmanFilter::update(const Ref<Vector<3>> z_w, const Ref<Vector<3>> z_c_0, const Ref<Matrix<3, 3>> R_C_W)
{
    if (!initialized_) throw std::runtime_error("Kalman filter is not initialized!");

    // Error covariance induced by sensor measurement in 3D space is computed
    // by propagating the measurement error covariance through the measurement model

    // Measurement in right camera frame
    Vector<3> z_c_1 = z_c_0 - Vector<3>(-0.12, 0.0, 0.0); // Baseline 0.12 m

    // Jacobian matrix of the measurement model with respect to the 3D position
    Matrix<3, 3> J_0 = computeJacobian(z_c_0);
    Matrix<3, 3> J_1 = computeJacobian(z_c_1);

    // Sensor noise
    Matrix<3, 3> R_0 = J_0 * D_ * J_0.transpose();
    Matrix<3, 3> R_1 = J_1 * D_ * J_1.transpose();

    // Bayesian fusion from left and right camera measurement
    Matrix<3, 3> R = R_0 * (R_0 + R_1).inverse() * R_1;

    // Change the orientation of covariance from camera frame to world
    R = R_C_W * R * R_C_W.transpose();


    // std::cout << "Q: \n" << Q_ << std::endl;
    // std::cout << "R: \n" << R << std::endl;


    // Innovation
    Matrix<3, 3> S = H_ * P_ * H_.transpose() + R;
    // Kalman gain
    K_ = P_ * H_.transpose() * S.inverse();
    // State update
    x_ = x_ + K_ * (z_w - H_ * x_);
    // Covariance update
    P_ = (I_ - K_ * H_) * P_;
}

Matrix<3, 3> KalmanFilter::computeJacobian(const Ref<Vector<3>> z) const
{
    Scalar x_c = z(0);
    Scalar y_c = z(1);
    Scalar z_c = z(2);

    Matrix<3, 3> J = (Matrix<3, 3>() << z_c,    0,  -x_c*z_c,
                                          0,  z_c,  -y_c*z_c,
                                          0,    0,  -z_c*z_c).finished(); // w.r.t. camera frame
    return J;
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
#include "flightlib/tracking_algorithm/kalman_filter_v1.hpp"

namespace flightlib {

KalmanFilterV1::KalmanFilterV1()
{

}

KalmanFilterV1::~KalmanFilterV1() {}

void KalmanFilterV1::reset()
{
    P_ = 1e2 * Matrix<9, 9>::Identity();
}

void KalmanFilterV1::init(const Scalar Ts, Ref<Vector<9>> x0, Ref<Matrix<9, 9>> P0, const Scalar sigma_w, const Scalar sigma_v,
                        const Scalar f, const Scalar c, const Scalar b)
{
    // Kalman filter sampling time
    Ts_ = Ts;

    // Initial guess
    x_ = x0;
    P_ = P0;

    useConstantAccModel();
    setSystemNoise(sigma_w);
    setSensorNoise(sigma_v);
    setCameraParameters(f, c, b);
}

void KalmanFilterV1::predict()
{
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Gamma_ * Q_ * Gamma_.transpose();
}

void KalmanFilterV1::update(const Ref<Vector<3>> y, const Ref<Vector<4>> pixel)
{
    H_ = computeJocobianH();

    // Residual
    Vector<3> nu = y - computeNonlinearFunch(x_[0], x_[3], x_[6]);

    // Innovation
    Matrix<3, 3> S = H_ * P_ * H_.transpose() + R_;
    // Kalman gain
    K_ = P_ * H_.transpose() * S.inverse();
    // State update
    x_ = x_ + K_ * nu;    
    // Covariance update
    P_ = (I_ - K_ * H_) * P_;
}

Vector<3> KalmanFilterV1::computeEstimatedPositionWrtWorld(Ref<Matrix<4, 4>> T_LC_W)
{
    // To homogeneous coordinates
    Vector<4> P_C(x_[0], x_[3], x_[6], 1);

    // Target coordinates w.r.t. body frame
    Vector<3> p_w = (T_LC_W * P_C).segment<3>(0);

    return p_w;
}

Scalar KalmanFilterV1::computeRangeWrtBody(Ref<Vector<3>> from, Ref<Matrix<4, 4>> T_LC_B)
{
    // To homogeneous coordinates
    Vector<4> P_C(x_[0], x_[3], x_[6], 1);

    // Target coordinates w.r.t. body frame
    Vector<3> p_b = (T_LC_B * P_C).segment<3>(0);

    Scalar range = sqrt(pow(p_b[0] - from[0], 2) + pow(p_b[1] - from[1], 2) + pow(p_b[2] - from[2], 2));

    return range;
}

void KalmanFilterV1::setSystemNoise(const Scalar sigma_w)
{
    // Define system noise matrix
    Q_ = Matrix<3, 3>::Identity() * pow(sigma_w, 2);
}

void KalmanFilterV1::setSensorNoise(const Scalar sigma_v)
{
    // Define sensor noise matrix
    R_ = Matrix<3, 3>::Identity() * pow(sigma_v, 2);
}

void KalmanFilterV1::setCameraParameters(const Scalar f, const Scalar c, const Scalar b)
{
    f_ = f;
    c_ = c;
    b_ = b;
}

void KalmanFilterV1::useConstantAccModel()
{
    // System matrix for constant acceleration model
    Matrix<3, 3> F_base = (Matrix<3, 3>() << 1, Ts_, 0.5 * pow(Ts_, 2),
                                             0,   1,               Ts_,
                                             0,   0,                 1).finished();

    F_.block<3,3>(0,0) = F_base;
    F_.block<3,3>(0,3) = Matrix<3, 3>::Zero();
    F_.block<3,3>(0,6) = Matrix<3, 3>::Zero();
    F_.block<3,3>(3,0) = Matrix<3, 3>::Zero();
    F_.block<3,3>(3,3) = F_base;
    F_.block<3,3>(3,6) = Matrix<3, 3>::Zero();
    F_.block<3,3>(6,0) = Matrix<3, 3>::Zero();
    F_.block<3,3>(6,3) = Matrix<3, 3>::Zero();
    F_.block<3,3>(6,6) = F_base;

    // Gamma.shape = (9, 3)
    Matrix<3, 1> Gamma_base = (Matrix<3, 1>() << pow(Ts_, 3) / 6, pow(Ts_, 2) / 2, Ts_).finished();
    Gamma_.block<3, 1>(0,0) = Gamma_base;
    Gamma_.block<3, 1>(0,1) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(0,2) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(3,0) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(3,1) = Gamma_base;
    Gamma_.block<3, 1>(3,2) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(6,0) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(6,1) = Matrix<3, 1>::Zero();
    Gamma_.block<3, 1>(6,2) = Gamma_base;
}

Vector<3> KalmanFilterV1::computeNonlinearFunch(const Scalar x, const Scalar y, const Scalar z) const
{
    return Vector<3>(x, y, z);
}

Matrix<3, 9> KalmanFilterV1::computeJocobianH() const
{
    Matrix<3, 9> H = (Matrix<3, 9>() << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 1, 0, 0).finished();
    return H;
}

}  // namespace flightlib
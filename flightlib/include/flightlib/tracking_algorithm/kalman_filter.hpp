#pragma once

#include <iostream>
#include <random>

#include "flightlib/common/types.hpp"


namespace flightlib {

/** @brief Extended Kalman filter with stereo camera sensor */
class KalmanFilter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KalmanFilter();
  ~KalmanFilter();

  void reset(void);
  void init(const Scalar Ts, Ref<Vector<6>> x0);

  // Prediction and correction
  void predict(void);
  void update(const Ref<Vector<3>> z, const Ref<Vector<3>> ego);

  // Sensor noise equation
  Scalar computeSensorNoise(const Scalar x);

  // Public get functions
  inline bool isInitialized(void) const { return initialized_; };
  inline Vector<3> getEstimatedPosition(void) const { return Vector<3>(x_[0], x_[2], x_[4]); };
  inline Vector<3> getEstimatedVelocity(void) const { return Vector<3>(x_[1], x_[3], x_[5]); };
  inline Matrix<6, 6> getErrorCovariance(void) const { return P_; };

  Matrix<3, 3> getPositionErrorCovariance(void) const;

 private:
  //
  bool initialized_;

  // From body to left camera
  Matrix<4, 4> T_B_C_;

  // Sampling time;
  Scalar Ts_;

  // Estimated state (w.r.t. left camera frame)
  Vector<6> x_;

  // Kalman filter matrix
  Matrix<6, 6> F_, P_;
  Matrix<3, 6> H_;
  Matrix<3, 3> Q_;
  Matrix<3, 3> R_;
  Matrix<6, 3> K_;

  //
  Matrix<6, 3> Gamma_;

  // System & sensor noise
  Scalar sigma_w_{7.0}, sigma_v_{2.0};
  // Adaptive sensor noise
  bool adaptive_{true};

  // Identity matrix
  Matrix<6, 6> I_ = Matrix<6, 6>::Identity();
};

}  // namespace flightlib

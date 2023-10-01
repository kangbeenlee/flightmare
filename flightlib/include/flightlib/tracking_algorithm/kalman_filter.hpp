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
  void update(const Ref<Vector<3>> z);

  // Public get functions
  inline Vector<3> getEstimatedPosition(void) const { return Vector<3>(x_[0], x_[2], x_[4]); };  
  inline Matrix<6, 6> getErrorCovariance(void) const { return P_; };
  inline bool isInitialized(void) const { return initialized_; };

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
  Scalar sigma_w_{1.0}, sigma_v_{5.0};

  // Identity matrix
  Matrix<6, 6> I_ = Matrix<6, 6>::Identity();
};

}  // namespace flightlib

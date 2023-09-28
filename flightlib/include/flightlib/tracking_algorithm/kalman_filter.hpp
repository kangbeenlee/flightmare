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
  void init(const Scalar Ts, Ref<Vector<9>> x0, const Scalar sigma_w, const Scalar sigma_v);

  // Prediction and correction
  void predict(void);
  void update(const Ref<Vector<3>> z);

  // Public get functions
  inline Vector<3> getEstimatedPosition(void) const { return Vector<3>(x_[0], x_[3], x_[6]); };  
  inline Matrix<9, 9> getErrorCovariance(void) const { return P_; };
  inline bool isInitialized(void) const { return initialized_; };

 private:
  //
  bool initialized_;

  // Sampling time;
  Scalar Ts_;

  // Estimated state (w.r.t. left camera frame)
  Vector<9> x_;

  // Kalman filter matrix
  Matrix<9, 9> F_, P_;
  Matrix<3, 9> H_;
  Matrix<3, 3> Q_;
  Matrix<3, 3> R_;
  Matrix<9, 3> K_;

  //
  Matrix<9, 3> Gamma_;

  // System and sensor noise
  Scalar sigma_w_{0.2};
  Scalar sigma_v_{30.0};

  // Identity matrix
  Matrix<9, 9> I_ = Matrix<9, 9>::Identity();
};

}  // namespace flightlib

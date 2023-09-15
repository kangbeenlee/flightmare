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
  void init(const Scalar Ts, Ref<Vector<9>> x0, Ref<Matrix<9, 9>> P0, const Scalar sigma_w, const Scalar sigma_v);

  // Prediction and correction
  void predict(void);
  void update(const Ref<Vector<3>> z);

  // Public get functions
  inline Vector<3> getEstimatedPositionWrtCamera(void) const { return Vector<3>(x_[0], x_[3], x_[6]); }; // w.r.t left camera
  inline Vector<9> getEstimatedStateWrtCamera(void) const { return x_; }; // w.r.t left camera
  inline Scalar getEsitmatedRangeWrtCamera(void) const { return pow(x_[0], 2) + pow(x_[1], 2), + pow(x_[2], 2); };
  inline Matrix<9, 9> getErrorCovariance(void) const { return P_; };

  // Public compute functions
  Vector<3> computeEstimatedPositionWrtWorld(Ref<Matrix<4, 4>> T_LC_W);
  Vector<3> computeEstimatedVelocityWrtWorld(Ref<Matrix<4, 4>> T_LC_W);
  Scalar computeRangeWrtBody(Ref<Vector<3>> from, Ref<Matrix<4, 4>> T_LC_B); // compute range from tracker to target

 private:
  //
  bool initialized_;

  // Sampling time;
  Scalar Ts_;

  // Estimated state (w.r.t. left camera frame)
  Vector<9> x_, x0_;

  // Kalman filter matrix
  Matrix<9, 9> F_, P_, P0_;
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

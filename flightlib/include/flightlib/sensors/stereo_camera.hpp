#pragma once

#include <iostream>
#include <random>

#include "flightlib/common/types.hpp"

namespace flightlib {

class StereoCamera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StereoCamera();
  ~StereoCamera();
  
  void init(const Ref<Vector<3>> B_r_LC, const Ref<Vector<3>> B_r_RC, const Ref<Matrix<3, 3>> R_B_C);

  // Public sensor data processing code
  bool computePixelPoint(const Ref<Vector<3>> position, const Ref<Matrix<4, 4>> T_W_B);

  //
  void reset(void);

  // Public get functions (sensor measurement)
  inline Vector<4> getGtPixels(void) const { return gt_pixels_; };
  inline Vector<4> getPixels(void) const { return pixels_; };
  inline Vector<3> getSensorMeasurement(void) const { return p_c_; };
  // inline Vector<3> getObjectPosition(void) const { return p_w_; };

  inline Matrix<4, 4> getFromLeftCameraToBody(void) const { return T_LC_B_; };

 private:
  bool initialized_{false};

  // From body to left camera
  Matrix<4, 4> T_B_LC_;
  // From body to right camera
  Matrix<4, 4> T_B_RC_;
  // From left camera to body
  Matrix<4, 4> T_LC_B_;

  // Intrinsic parameters of ZED X mini (HD1080 resolutoin, 0.003mm x 0.003mm pixel size)
  Scalar h_fov_{110.0};
  Scalar v_fov_{80.0};
  int f_{733}; // Focal length (pixel), 2.2 (mm)
  int c_x_{960};
  int c_y_{540};
  int width_{1920};
  int height_{1080};
  Matrix<3, 3> K_; // Intrinsic matrix

  // Stereo camera parameter
  Scalar b_{0.12}; // Baseline

  // Stereo camera sensor measurement
  Vector<4> gt_pixels_;
  Vector<4> pixels_;
  Vector<3> p_c_; // Sensor measurement
  // Vector<3> p_w_; // object position based on world frame

  // Random variable generator for pixel noise
  std::normal_distribution<Scalar> norm_dist_{0.0, 0.2};
  std::random_device rd_;
  std::mt19937 random_gen_{rd_()};
};

}  // namespace flightlib

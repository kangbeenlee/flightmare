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
  
  // Public sensor data processing code
  bool processImagePoint(const Ref<Vector<3>> target_point, const Ref<Matrix<4, 4>> T_W_B);

  //
  void reset(void);

  // Public get functions (sensor measurement)
  inline Vector<4> getGroundTruthPixels(void) const { return gt_pixels_; };
  inline Vector<4> getPixels(void) const { return pixels_; };
  inline Scalar getGroundTruthDepth(void) const { return gt_depth_; };
  inline Scalar getDepth(void) const { return depth_; };
  inline Vector<3> getGroundTruthPosition(void) const { return gt_p_c_; };
  inline Vector<3> getTargetPosition(void) const { return p_c_; };

  // Public get functions
  inline Scalar getFOV(void) const { return fov_; };
  inline Scalar getFocalLength(void) const { return f_; };
  inline Scalar getPrincipalPoint(void) const { return c_; };
  inline Scalar getBaseline(void) const { return b_; };
  inline int getWidth(void) const { return width_; };
  inline int getHeight(void) const { return height_; };

  inline Matrix<4, 4> getFromLeftCameraToBody(void) const { return T_LC_B_; };

 private:
  // From body to left camera
  Matrix<3, 3> R_B_LC_; // Rotation
  Vector<3> B_r_LC_; // Translation
  Matrix<4, 4> T_B_LC_; // Transformation
  // From body to right camera
  Matrix<3, 3> R_B_RC_;
  Vector<3> B_r_RC_;
  Matrix<4, 4> T_B_RC_;

  // From left camera to body
  Matrix<4, 4> T_LC_B_;

  // Intrinsic parameters of left and right cameras
  Scalar fov_{90.0};
  int f_{320}; // Focal length (pixel)
  int c_{320}; // Principal point (pixel)
  int width_{640};
  int height_{640};
  Matrix<3, 3> K_; // Intrinsic matrix

  // Stereo camera parameter
  Scalar b_{0.12}; // Baseline

  // Stereo camera sensor measurement
  Vector<4> gt_pixels_;
  Vector<4> pixels_;
  Scalar gt_depth_;
  Scalar depth_;
  Vector<3> gt_p_c_, p_c_;

  // Random variable generator for pixel noise
  std::normal_distribution<Scalar> norm_dist_{0.0, 0.2};
  std::random_device rd_;
  std::mt19937 random_gen_{rd_()};
};

}  // namespace flightlib

#pragma once

#include <stdlib.h>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <valarray>
#include <algorithm>
#include <fstream>
#include <cmath>

// #include "flightlib/common/types.hpp"

namespace flightlib {

class MinimumSnapTrajectory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MinimumSnapTrajectory();
  ~MinimumSnapTrajectory();

  Eigen::VectorXf getPoly(int order, int der, float t);
  Eigen::VectorXf minTraj(Eigen::VectorXf waypoints, Eigen::VectorXf times, int order, int time_step);
  Eigen::VectorXf posWayPointMin(float time, int time_step, Eigen::VectorXf t_wps, int order, Eigen::VectorXf cX, Eigen::VectorXf cY, Eigen::VectorXf cZ);
  
  void initTrajectory();
  void setMinimumSnapTrajectory(const Eigen::MatrixXf& way_points, const Eigen::VectorXf& segment_times);
  Eigen::VectorXf getDesiredPosVelAcc(float time);

 private:
  int order_{8};
  int time_step_;
  Eigen::VectorXf way_point_times_;
  int total_time_;
  Eigen::VectorXf coeff_x_, coeff_y_, coeff_z_;
};

}  // namespace flightlib
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
  Eigen::VectorXf posWayPointMin(float time, int time_step, Eigen::VectorXf t_wps, int order,
                                 Eigen::VectorXf cX, Eigen::VectorXf cY, Eigen::VectorXf cZ);
};

}  // namespace flightlib
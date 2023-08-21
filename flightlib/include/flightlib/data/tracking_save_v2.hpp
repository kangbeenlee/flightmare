#pragma once

#include <stdlib.h>
#include <armadillo>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class TrackingSaveV2 {
 public:
  TrackingSaveV2() {};
  ~TrackingSaveV2() {};

  void store(const Vector<3> gt_position,
             const Vector<3> estim_position,
             const Matrix<9, 9> covariance,
             const Scalar sim_dt)
  {
    gt_data_(0, i) = gt_position[0];
    gt_data_(1, i) = gt_position[1];
    gt_data_(2, i) = gt_position[2];
    estim_data_(0, i) = estim_position[0];
    estim_data_(1, i) = estim_position[1];
    estim_data_(2, i) = estim_position[2];
    error_covariance_(0, i) = covariance(0, 0);
    error_covariance_(1, i) = covariance(3, 3);
    error_covariance_(2, i) = covariance(6, 6);
    time_[i] = t;
    
    i++;
    t += sim_dt;
  };

  void save() {
    // Save to files (for visualization)
    arma::mat gt_x = gt_data_.row(0);
    arma::mat gt_y = gt_data_.row(1);
    arma::mat gt_z = gt_data_.row(2);
    gt_x.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_x.txt", arma::raw_ascii);
    gt_y.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_y.txt", arma::raw_ascii);
    gt_z.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/gt_z.txt", arma::raw_ascii);

    arma::mat estim_x = estim_data_.row(0);
    arma::mat estim_y = estim_data_.row(1);
    arma::mat estim_z = estim_data_.row(2);
    estim_x.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_x.txt", arma::raw_ascii);
    estim_y.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_y.txt", arma::raw_ascii);
    estim_z.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/estim_z.txt", arma::raw_ascii);

    arma::mat cov_x = error_covariance_.row(0);
    arma::mat cov_y = error_covariance_.row(1);
    arma::mat cov_z = error_covariance_.row(2);
    cov_x.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_x.txt", arma::raw_ascii);
    cov_y.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_y.txt", arma::raw_ascii);
    cov_z.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/cov_z.txt", arma::raw_ascii);

    time_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/time.txt", arma::raw_ascii);
  }

  bool isFull() {
    if (i == buffer_) return true;
    else return false;
  }

  void reset() {
    i = 0;
    t = 0.0;
    gt_data_ = arma::zeros(state_size_, buffer_);
    estim_data_ = arma::zeros(state_size_, buffer_);
    error_covariance_ = arma::zeros(state_size_, buffer_);
    time_ = arma::zeros(buffer_);
  }

 private:
  unsigned int i = 0;
  Scalar t = 0.0;
  unsigned int state_size_ = 3; // x, y, z
  unsigned int buffer_ = 500; // time buffer
  unsigned int measurement_size_ = 4;
  arma::mat gt_data_ = arma::zeros(state_size_, buffer_);
  arma::mat estim_data_ = arma::zeros(state_size_, buffer_);
  arma::mat error_covariance_ = arma::zeros(state_size_, buffer_);
  arma::mat time_ = arma::zeros(buffer_);
};

}  // namespace flightlib

#pragma once

#include <stdlib.h>
#include <armadillo>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class SensorSaveV1 {
 public:
  SensorSaveV1() {};
  ~SensorSaveV1() {};

  void store(Ref<Vector<4>> gt_measurement, Ref<Vector<4>> measurement, const Scalar gt_depth, const Scalar depth, const Scalar sim_dt)
  {
    gt_data_(0, i) = gt_measurement[0];
    gt_data_(1, i) = gt_measurement[1];
    gt_data_(2, i) = gt_measurement[2];
    gt_data_(3, i) = gt_measurement[3];
    sensor_output_(0, i) = measurement[0];
    sensor_output_(1, i) = measurement[1];
    sensor_output_(2, i) = measurement[2];
    sensor_output_(3, i) = measurement[3];
    gt_depth_(i) = gt_depth;
    depth_(i) = depth;

    time_[i] = t;
    
    i++;
    t += sim_dt;
  };

  void save() {
    // Save to files (for visualization)
    arma::mat gt_u_l = gt_data_.row(0);
    arma::mat gt_v_l = gt_data_.row(1);
    arma::mat gt_u_r = gt_data_.row(2);
    arma::mat gt_v_r = gt_data_.row(3);
    gt_u_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_l.txt", arma::raw_ascii);
    gt_v_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_l.txt", arma::raw_ascii);
    gt_u_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_r.txt", arma::raw_ascii);
    gt_v_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_r.txt", arma::raw_ascii);

    arma::mat u_l = sensor_output_.row(0);
    arma::mat v_l = sensor_output_.row(1);
    arma::mat u_r = sensor_output_.row(2);
    arma::mat v_r = sensor_output_.row(3);
    u_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_l.txt", arma::raw_ascii);
    v_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_l.txt", arma::raw_ascii);
    u_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_r.txt", arma::raw_ascii);
    v_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_r.txt", arma::raw_ascii);

    gt_depth_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_depth.txt", arma::raw_ascii);
    depth_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/depth.txt", arma::raw_ascii);

    time_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/time.txt", arma::raw_ascii);
  }

  bool isFull() {
    if (i == buffer_) return true;
    else return false;
  }

  void reset() {
    i = 0;
    t = 0.0;
    gt_data_ = arma::zeros(output_size_, buffer_);
    sensor_output_ = arma::zeros(output_size_, buffer_);
    gt_depth_ = arma::zeros(buffer_);
    depth_ = arma::zeros(buffer_);
    time_ = arma::zeros(buffer_);
  }

 private:
  unsigned int i = 0;
  Scalar t = 0.0;
  unsigned int output_size_ = 4; // x, y, z, range
  unsigned int buffer_ = 500; // time buffer
  arma::mat gt_data_ = arma::zeros(output_size_, buffer_);
  arma::mat sensor_output_ = arma::zeros(output_size_, buffer_);
  arma::mat gt_depth_ = arma::zeros(buffer_);
  arma::mat depth_ = arma::zeros(buffer_);
  arma::mat time_ = arma::zeros(buffer_);
};

}  // namespace flightlib

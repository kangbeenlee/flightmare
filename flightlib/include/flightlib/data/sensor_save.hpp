#pragma once

#include <stdlib.h>
#include <armadillo>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class SensorSave {
 public:
  SensorSave() {};
  ~SensorSave() {};

  void store(Ref<Vector<4>> gt_pixel, Ref<Vector<4>> pixel, Ref<Vector<3>> gt_measurement, Ref<Vector<3>> measurement, const Scalar sim_dt)
  {
    gt_pixel_(0, i) = gt_pixel[0];
    gt_pixel_(1, i) = gt_pixel[1];
    gt_pixel_(2, i) = gt_pixel[2];
    gt_pixel_(3, i) = gt_pixel[3];
    pixel_(0, i) = pixel[0];
    pixel_(1, i) = pixel[1];
    pixel_(2, i) = pixel[2];
    pixel_(3, i) = pixel[3];
    gt_data_(0, i) = gt_measurement[0];
    gt_data_(1, i) = gt_measurement[1];
    gt_data_(2, i) = gt_measurement[2];
    data_(0, i) = measurement[0];
    data_(1, i) = measurement[1];
    data_(2, i) = measurement[2];

    time_[i] = t;
    
    i++;
    t += sim_dt;
  };

  void save() {
    // Save to files (for visualization)
    arma::mat gt_u_l = gt_pixel_.row(0);
    arma::mat gt_v_l = gt_pixel_.row(1);
    arma::mat gt_u_r = gt_pixel_.row(2);
    arma::mat gt_v_r = gt_pixel_.row(3);
    gt_u_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_l.txt", arma::raw_ascii);
    gt_v_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_l.txt", arma::raw_ascii);
    gt_u_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_u_r.txt", arma::raw_ascii);
    gt_v_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_v_r.txt", arma::raw_ascii);

    arma::mat u_l = pixel_.row(0);
    arma::mat v_l = pixel_.row(1);
    arma::mat u_r = pixel_.row(2);
    arma::mat v_r = pixel_.row(3);
    u_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_l.txt", arma::raw_ascii);
    v_l.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_l.txt", arma::raw_ascii);
    u_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/u_r.txt", arma::raw_ascii);
    v_r.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/v_r.txt", arma::raw_ascii);

    arma::mat gt_x = gt_data_.row(0);
    arma::mat gt_y = gt_data_.row(1);
    arma::mat gt_z = gt_data_.row(2);
    gt_x.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_x.txt", arma::raw_ascii);
    gt_y.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_y.txt", arma::raw_ascii);
    gt_z.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/gt_z.txt", arma::raw_ascii);

    arma::mat x = data_.row(0);
    arma::mat y = data_.row(1);
    arma::mat z = data_.row(2);
    x.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/x.txt", arma::raw_ascii);
    y.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/y.txt", arma::raw_ascii);
    z.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/z.txt", arma::raw_ascii);

    time_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/sensor_output/time.txt", arma::raw_ascii);
  }

  bool isFull() {
    if (i == buffer_) return true;
    else return false;
  }

  void reset() {
    i = 0;
    t = 0.0;
    gt_pixel_ = arma::zeros(pixel_size_, buffer_);
    pixel_ = arma::zeros(pixel_size_, buffer_);
    gt_data_ = arma::zeros(output_size_, buffer_);
    data_ = arma::zeros(output_size_, buffer_);
    time_ = arma::zeros(buffer_);
  }

 private:
  unsigned int i = 0;
  Scalar t = 0.0;
  unsigned int output_size_ = 3; // x, y, z, range
  unsigned int pixel_size_ = 4;
  unsigned int buffer_ = 500; // time buffer
  arma::mat gt_pixel_ = arma::zeros(pixel_size_, buffer_);
  arma::mat pixel_ = arma::zeros(pixel_size_, buffer_);
  arma::mat gt_data_ = arma::zeros(output_size_, buffer_);
  arma::mat data_ = arma::zeros(output_size_, buffer_);
  arma::mat time_ = arma::zeros(buffer_);
};

}  // namespace flightlib

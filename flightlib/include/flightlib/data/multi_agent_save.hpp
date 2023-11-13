#pragma once

#include <stdlib.h>
#include <armadillo>
#include <string>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class MultiAgentSave {
 public:
  MultiAgentSave() {};
  ~MultiAgentSave() {};

  void init(const int num_targets) {
    initialized_ = true;
    i = 0;
    t = 0.0;
    num_targets_ = num_targets;
    state_size_ = 3; // x, y, z
    cov_size_ = 9;
    buffer_ = 1000; // time buffer

    for (int k = 0; k < num_targets_; ++k) {
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    } 
    time_ = arma::zeros(buffer_);
  }

  void reset() {
    i = 0;
    t = 0.0;
    for (int k = 0; k < num_targets_; ++k) {
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    } 
    time_ = arma::zeros(buffer_);
  }

  void store(const std::vector<Vector<3>> target_estim_position,
             const std::vector<Matrix<3, 3>> target_covariance,
             const Scalar sim_dt)
  {
    if (!initialized_) throw std::runtime_error("Tracking Recorder is not initialized!");

    for (int k = 0; k < num_targets_; ++k) {
      target_estim_data_[k](0, i) = target_estim_position[k][0];
      target_estim_data_[k](1, i) = target_estim_position[k][1];
      target_estim_data_[k](2, i) = target_estim_position[k][2];

      target_error_cov_[k](0, i) = target_covariance[k](0, 0); // sigma_xx
      target_error_cov_[k](1, i) = target_covariance[k](0, 1); // sigma_xy
      target_error_cov_[k](2, i) = target_covariance[k](0, 2); // sigma_xz
      target_error_cov_[k](3, i) = target_covariance[k](1, 0); // sigma_yx
      target_error_cov_[k](4, i) = target_covariance[k](1, 1); // sigma_yy
      target_error_cov_[k](5, i) = target_covariance[k](1, 2); // sigma_yz
      target_error_cov_[k](6, i) = target_covariance[k](2, 0); // sigma_zx
      target_error_cov_[k](7, i) = target_covariance[k](2, 1); // sigma_zy
      target_error_cov_[k](8, i) = target_covariance[k](2, 2); // sigma_zz
    }
    
    time_[i] = t;
    
    i++;
    t += sim_dt;
  };

  void save() {
    // Save to files (for visualization)
    std::string save_path = "/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/multi/";

    for (int k = 0; k < num_targets_; ++k) {
      arma::mat target_estim_x = target_estim_data_[k].row(0);
      arma::mat target_estim_y = target_estim_data_[k].row(1);
      arma::mat target_estim_z = target_estim_data_[k].row(2);
      target_estim_x.save(save_path+"target_estim_x_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_estim_y.save(save_path+"target_estim_y_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_estim_z.save(save_path+"target_estim_z_"+std::to_string(k)+".txt", arma::raw_ascii);

      arma::mat target_cov_xx = target_error_cov_[k].row(0);
      arma::mat target_cov_xy = target_error_cov_[k].row(1);
      arma::mat target_cov_xz = target_error_cov_[k].row(2);
      arma::mat target_cov_yx = target_error_cov_[k].row(3);
      arma::mat target_cov_yy = target_error_cov_[k].row(4);
      arma::mat target_cov_yz = target_error_cov_[k].row(5);
      arma::mat target_cov_zx = target_error_cov_[k].row(6);
      arma::mat target_cov_zy = target_error_cov_[k].row(7);
      arma::mat target_cov_zz = target_error_cov_[k].row(8);
      target_cov_xx.save(save_path+"target_cov_xx_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_xy.save(save_path+"target_cov_xy_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_xz.save(save_path+"target_cov_xz_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_yx.save(save_path+"target_cov_yx_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_yy.save(save_path+"target_cov_yy_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_yz.save(save_path+"target_cov_yz_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_zx.save(save_path+"target_cov_zx_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_zy.save(save_path+"target_cov_zy_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_zz.save(save_path+"target_cov_zz_"+std::to_string(k)+".txt", arma::raw_ascii);
    }

    time_.save(save_path+"time.txt", arma::raw_ascii);
  }

  bool isFull() {
    if (i == buffer_) return true;
    else return false;
  }

 private:
  bool initialized_{false};
  int i;
  Scalar t;
  int num_targets_;
  int state_size_; // x, y, z
  int cov_size_; // x, y, z
  int buffer_; // time buffer
  std::vector<arma::mat> target_estim_data_;
  std::vector<arma::mat> target_error_cov_;
  arma::mat time_;
};

}  // namespace flightlib

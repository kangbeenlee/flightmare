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
    buffer_ = 600; // time buffer

    for (int k = 0; k < num_targets_; ++k) {
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
    } 
    time_ = arma::zeros(buffer_);
  }

  void reset() {
    i = 0;
    t = 0.0;
    for (int k = 0; k < num_targets_; ++k) {
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
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
      target_error_cov_[k](0, i) = target_covariance[k](0, 0);
      target_error_cov_[k](1, i) = target_covariance[k](1, 1);
      target_error_cov_[k](2, i) = target_covariance[k](2, 2);
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

      arma::mat target_cov_x = target_error_cov_[k].row(0);
      arma::mat target_cov_y = target_error_cov_[k].row(1);
      arma::mat target_cov_z = target_error_cov_[k].row(2);
      target_cov_x.save(save_path+"target_cov_x_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_y.save(save_path+"target_cov_y_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_cov_z.save(save_path+"target_cov_z_"+std::to_string(k)+".txt", arma::raw_ascii);
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
  int buffer_; // time buffer
  std::vector<arma::mat> target_estim_data_;
  std::vector<arma::mat> target_error_cov_;
  arma::mat time_;
};

}  // namespace flightlib

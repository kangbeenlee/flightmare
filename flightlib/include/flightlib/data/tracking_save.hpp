#pragma once

#include <stdlib.h>
#include <armadillo>
#include <string>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class TrackingSave {
 public:
  TrackingSave() {};
  ~TrackingSave() {};

  void init(const int num_targets, const int num_trackers, const int agent_id) {
    agent_id_ = agent_id;
    initialized_ = true;
    i = 0;
    t = 0.0;
    num_targets_ = num_targets;
    num_trackers_ = num_trackers;
    state_size_ = 3; // x, y, z
    cov_size_ = 9;
    buffer_ = 600; // time buffer

    for (int k = 0; k < num_targets_; ++k) {
      target_gt_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    } 
    for (int k = 0; k < num_trackers_; ++k) {
      tracker_gt_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      tracker_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      tracker_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    }

    ego_state_size_ = 7;
    ego_data_ = arma::zeros(ego_state_size_, buffer_);
    time_ = arma::zeros(buffer_);
  }

  void reset() {
    i = 0;
    t = 0.0;
    for (int k = 0; k < num_targets_; ++k) {
      target_gt_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      target_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    } 
    for (int k = 0; k < num_trackers_; ++k) {
      tracker_gt_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      tracker_estim_data_.push_back(arma::zeros<arma::mat>(state_size_, buffer_));
      tracker_error_cov_.push_back(arma::zeros<arma::mat>(cov_size_, buffer_));
    }
    ego_data_ = arma::zeros(ego_state_size_, buffer_);
    time_ = arma::zeros(buffer_);
  }

  void store(const Vector<3> ego_position,
             const Vector<4> ego_orientation,
             const std::vector<Vector<3>> target_gt_position,
             const std::vector<Vector<3>> target_estim_position,
             const std::vector<Matrix<3, 3>> target_covariance,
             const std::vector<Vector<3>> tracker_gt_position,
             const std::vector<Vector<3>> tracker_estim_position,
             const std::vector<Matrix<3, 3>> tracker_covariance,
             const Scalar sim_dt)
  {
    if (!initialized_) throw std::runtime_error("Tracking Recorder is not initialized!");

    ego_data_(0, i) = ego_position[0];
    ego_data_(1, i) = ego_position[1];
    ego_data_(2, i) = ego_position[2];
    ego_data_(3, i) = ego_orientation[0];
    ego_data_(4, i) = ego_orientation[1];
    ego_data_(5, i) = ego_orientation[2];
    ego_data_(6, i) = ego_orientation[3];

    for (int k = 0; k < num_targets_; ++k) {
      target_gt_data_[k](0, i) = target_gt_position[k][0];
      target_gt_data_[k](1, i) = target_gt_position[k][1];
      target_gt_data_[k](2, i) = target_gt_position[k][2];
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

    for (int k = 0; k < num_trackers_; ++k) {
      tracker_gt_data_[k](0, i) = tracker_gt_position[k][0];
      tracker_gt_data_[k](1, i) = tracker_gt_position[k][1];
      tracker_gt_data_[k](2, i) = tracker_gt_position[k][2];
      tracker_estim_data_[k](0, i) = tracker_estim_position[k][0];
      tracker_estim_data_[k](1, i) = tracker_estim_position[k][1];
      tracker_estim_data_[k](2, i) = tracker_estim_position[k][2];

      tracker_error_cov_[k](0, i) = tracker_covariance[k](0, 0); // sigma_xx
      tracker_error_cov_[k](1, i) = tracker_covariance[k](0, 1); // sigma_xy
      tracker_error_cov_[k](2, i) = tracker_covariance[k](0, 2); // sigma_xz
      tracker_error_cov_[k](3, i) = tracker_covariance[k](1, 0); // sigma_yx
      tracker_error_cov_[k](4, i) = tracker_covariance[k](1, 1); // sigma_yy
      tracker_error_cov_[k](5, i) = tracker_covariance[k](1, 2); // sigma_yz
      tracker_error_cov_[k](6, i) = tracker_covariance[k](2, 0); // sigma_zx
      tracker_error_cov_[k](7, i) = tracker_covariance[k](2, 1); // sigma_zy
      tracker_error_cov_[k](8, i) = tracker_covariance[k](2, 2); // sigma_zz
    }
    
    time_[i] = t;
    
    i++;
    t += sim_dt;
  };

  void save() {
    // Save to files (for visualization)
    arma::mat ego_x = ego_data_.row(0);
    arma::mat ego_y = ego_data_.row(1);
    arma::mat ego_z = ego_data_.row(2);
    arma::mat ego_qw = ego_data_.row(3);
    arma::mat ego_qx = ego_data_.row(4);
    arma::mat ego_qy = ego_data_.row(5);
    arma::mat ego_qz = ego_data_.row(6);
    
    std::string save_path = "/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/tracking_output/";
    save_path = save_path + "tracker_" + std::to_string(agent_id_) + "/";

    ego_x.save(save_path+"ego_x.txt", arma::raw_ascii);
    ego_y.save(save_path+"ego_y.txt", arma::raw_ascii);
    ego_z.save(save_path+"ego_z.txt", arma::raw_ascii);
    ego_qw.save(save_path+"ego_qw.txt", arma::raw_ascii);
    ego_qx.save(save_path+"ego_qx.txt", arma::raw_ascii);
    ego_qy.save(save_path+"ego_qy.txt", arma::raw_ascii);
    ego_qz.save(save_path+"ego_qz.txt", arma::raw_ascii);

    for (int k = 0; k < num_targets_; ++k) {
      arma::mat target_gt_x = target_gt_data_[k].row(0);
      arma::mat target_gt_y = target_gt_data_[k].row(1);
      arma::mat target_gt_z = target_gt_data_[k].row(2);
      target_gt_x.save(save_path+"target_gt_x_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_gt_y.save(save_path+"target_gt_y_"+std::to_string(k)+".txt", arma::raw_ascii);
      target_gt_z.save(save_path+"target_gt_z_"+std::to_string(k)+".txt", arma::raw_ascii);
      
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

    for (int k = 0; k < num_trackers_; ++k) {
      arma::mat tracker_gt_x = tracker_gt_data_[k].row(0);
      arma::mat tracker_gt_y = tracker_gt_data_[k].row(1);
      arma::mat tracker_gt_z = tracker_gt_data_[k].row(2);
      tracker_gt_x.save(save_path+"tracker_gt_x_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_gt_y.save(save_path+"tracker_gt_y_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_gt_z.save(save_path+"tracker_gt_z_"+std::to_string(k)+".txt", arma::raw_ascii);
      
      arma::mat tracker_estim_x = tracker_estim_data_[k].row(0);
      arma::mat tracker_estim_y = tracker_estim_data_[k].row(1);
      arma::mat tracker_estim_z = tracker_estim_data_[k].row(2);
      tracker_estim_x.save(save_path+"tracker_estim_x_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_estim_y.save(save_path+"tracker_estim_y_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_estim_z.save(save_path+"tracker_estim_z_"+std::to_string(k)+".txt", arma::raw_ascii);


      arma::mat tracker_cov_xx = tracker_error_cov_[k].row(0);
      arma::mat tracker_cov_xy = tracker_error_cov_[k].row(1);
      arma::mat tracker_cov_xz = tracker_error_cov_[k].row(2);
      arma::mat tracker_cov_yx = tracker_error_cov_[k].row(3);
      arma::mat tracker_cov_yy = tracker_error_cov_[k].row(4);
      arma::mat tracker_cov_yz = tracker_error_cov_[k].row(5);
      arma::mat tracker_cov_zx = tracker_error_cov_[k].row(6);
      arma::mat tracker_cov_zy = tracker_error_cov_[k].row(7);
      arma::mat tracker_cov_zz = tracker_error_cov_[k].row(8);
      tracker_cov_xx.save(save_path+"tracker_cov_xx_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_xy.save(save_path+"tracker_cov_xy_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_xz.save(save_path+"tracker_cov_xz_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_yx.save(save_path+"tracker_cov_yx_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_yy.save(save_path+"tracker_cov_yy_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_yz.save(save_path+"tracker_cov_yz_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_zx.save(save_path+"tracker_cov_zx_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_zy.save(save_path+"tracker_cov_zy_"+std::to_string(k)+".txt", arma::raw_ascii);
      tracker_cov_zz.save(save_path+"tracker_cov_zz_"+std::to_string(k)+".txt", arma::raw_ascii);
    }

    time_.save(save_path+"time.txt", arma::raw_ascii);
  }

  bool isFull() {
    if (i == buffer_) return true;
    else return false;
  }

 private:
  int agent_id_;
  bool initialized_{false};
  int i;
  Scalar t;
  int num_targets_;
  int num_trackers_;
  int state_size_; // x, y, z
  int cov_size_;
  int buffer_; // time buffer
  std::vector<arma::mat> target_gt_data_;
  std::vector<arma::mat> target_estim_data_;
  std::vector<arma::mat> target_error_cov_;
  std::vector<arma::mat> tracker_gt_data_;
  std::vector<arma::mat> tracker_estim_data_;
  std::vector<arma::mat> tracker_error_cov_;
  int ego_state_size_; // x, y, z, qw, qx, qy, qz
  arma::mat ego_data_;
  arma::mat time_;
};

}  // namespace flightlib
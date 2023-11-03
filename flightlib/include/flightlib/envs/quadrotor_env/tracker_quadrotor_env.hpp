#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/tracker_quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

// #include <opencv2/opencv.hpp>

// Header files for target tracking
#include "flightlib/sensors/stereo_camera.hpp"
#include "flightlib/tracking_algorithm/kalman_filter.hpp"
#include "flightlib/tracking_algorithm/hungarian.hpp"
#include "flightlib/data/sensor_save.hpp"
#include "flightlib/data/tracking_save.hpp"

namespace flightlib {

namespace trackerquadenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 3,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kNObs = 12,
  // control actions
  kAct = 0,
  kNAct = 4,
  };
};
class TrackerQuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackerQuadrotorEnv();
  TrackerQuadrotorEnv(const std::string &cfg_path);
  ~TrackerQuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  bool reset(Ref<Vector<>> obs, Ref<Vector<>> position,
             const std::vector<Vector<3>>& target_positions, const std::vector<Vector<3>>& other_tracker_positions, const int agent_id);

  //
  Matrix<3, 3> Rot_x(const Scalar angle) const;
  Matrix<3, 3> Rot_y(const Scalar angle) const;
  Matrix<3, 3> Rot_z(const Scalar angle) const;
  Matrix<3, 3> eulerToRotation(const Ref<Vector<3>> euler_zyx) const;
  Vector<4> eulerToQuaternion(const Ref<Vector<3>> euler_zyx) const;
  Matrix<4, 4> getBodyToWorld(void) const;

  //
  Vector<3> fromWorldToCamera(Vector<3> point_w) const;
  Vector<3> fromCameraToWorld(Vector<3> point_c) const;
  Matrix<3, 3> fromCameraToWorld(Matrix<3, 3> cov) const;
  Matrix<3, 3> rotateFromCameraToWorld(void) const;


  //
  Vector<3> getPosition(void) const;

  // Test for only tracker without target tracking
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;  // Not used
  Scalar trackerStep(const Ref<Vector<>> act, Ref<Vector<>> obs,
                     const std::vector<Vector<3>>& target_positions, const std::vector<Vector<3>>& other_tracker_positions);

  // Reward function for RL
  Scalar rewardFunction();

  //
  Scalar computeEuclideanDistance(Ref<Vector<3>> p1, Ref<Vector<3>> p2);

  //
  inline Vector<3> getEstimatedTargetPosition(const int i) { return target_kalman_filters_[i]->getEstimatedPosition(); };
  inline Matrix<3, 3> getTargetPositionCov(const int i) { return target_kalman_filters_[i]->getPositionErrorCovariance(); };
  inline Scalar getTargetPositionCovNorm(const int i) { return (target_kalman_filters_[i]->getPositionErrorCovariance()).norm(); };
  inline Scalar getTargetPositionCovDet(const int i) { return (target_kalman_filters_[i]->getPositionErrorCovariance()).determinant(); };


  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

 private:
  // Quadrotor
  int agent_id_;
  Scalar radius_{0.25};
  std::shared_ptr<TrackerQuadrotor> tracker_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"TrackerQaudrotorEnv"};

  // Stereo camera
  int num_cameras_;
  std::vector<std::shared_ptr<StereoCamera>> multi_stereo_;
  Vector<3> d_l_;
  Vector<3> d_r_;
  Matrix<3, 3> R_B_C_; // Frame rotation matrix
  Matrix<4, 4> T_B_LC_; // Frame translation matrix
  Matrix<4, 4> T_LC_B_; // Frame translation matrix

  // Kalman filter
  std::vector<std::shared_ptr<KalmanFilter>> target_kalman_filters_, tracker_kalman_filters_;
  int num_targets_, num_trackers_; // except tracker itself

  // Sensor data recoder
  std::shared_ptr<SensorSave> sensor_save_;
  bool sensor_flag_{true};
  Vector<4> gt_pixels_, pixels_;
  Vector<3> measured_position_;

  // Tracking data recoder
  std::shared_ptr<TrackingSave> tracking_save_;
  bool tracking_flag_{true};
  std::vector<Vector<3>> gt_target_positions_, estimated_target_positions_, estimated_target_velocities_;
  std::vector<Vector<3>> gt_tracker_positions_, estimated_tracker_positions_, estimated_tracker_velocities_;
  std::vector<Scalar> estimated_target_ranges_, estimated_tracker_ranges_;

  //
  bool first_{true};
  Scalar prev_range_;

  // PID controller for linear velocity (LV) control policy tracker
  Scalar kp_vxy_, ki_vxy_, kd_vxy_, kp_vz_, ki_vz_, kd_vz_, kp_angle_, ki_angle_, kd_angle_, kp_wz_, ki_wz_, kd_wz_;

  // RL reward
  Vector<4> prev_act_{0.0, 0.0, 0.0, 0.0};


  // Observations and actions (for RL)
  // Vector<trackerquadenv::kNObs> quad_obs_;
  // Vector<28> quad_obs_;
  // Vector<55> quad_obs_;
  Vector<73> quad_obs_;
  Vector<trackerquadenv::kNAct> quad_act_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib
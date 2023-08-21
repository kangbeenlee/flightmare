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
#include "flightlib/sensors/stereo_camera.hpp"
#include "flightlib/tracking_algorithm/kalman_filter_v1.hpp"
#include "flightlib/data/sensor_save_v1.hpp"
#include "flightlib/data/sensor_save_v2.hpp"
#include "flightlib/data/tracking_save_v1.hpp"
#include "flightlib/data/tracking_save_v2.hpp"

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
  TrackerQuadrotorEnv(const YAML::Node &cfg_path);
  ~TrackerQuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  // Test for only tracker without target tracking
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;
  Scalar trackerStep(const Ref<Vector<>> act, Ref<Vector<>> obs, Vector<3> target_point);

  //
  Vector<4> eulerToQuaternion(const Ref<Vector<3>> euler_zyx) const;
  Matrix<4, 4> getBodyToWorld() const;

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
  std::shared_ptr<TrackerQuadrotor> tracker_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"TrackerQaudrotorEnv"};

  // Stereo camera
  std::shared_ptr<StereoCamera> stereo_camera_;
  SensorSaveV2 sensor_save_;
  bool sensor_flag_{true};

  // Kalman filter
  std::shared_ptr<KalmanFilterV1> kf_;
  TrackingSaveV2 tracking_save_;
  Vector<3> gt_target_position_, target_position_;
  Vector<4> gt_pixels_, pixels_;
  bool tracking_flag_{true};
  
  // PID controller
  Scalar kp_vxy_, ki_vxy_, kd_vxy_, kp_vz_, ki_vz_, kd_vz_, kp_angle_, ki_angle_, kd_angle_, kp_wz_, ki_wz_, kd_wz_;

  // Observations and actions (for RL)
  Vector<trackerquadenv::kNObs> quad_obs_;
  Vector<trackerquadenv::kNAct> quad_act_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib
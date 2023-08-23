#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/custom_command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/target_quadrotor.hpp"
#include "flightlib/trajectory_planner/minimum_snap_trajectory.hpp"



namespace flightlib {

namespace targetquadenv {

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
class TargetQuadrotorEnv final : public EnvBase{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TargetQuadrotorEnv();
  TargetQuadrotorEnv(const std::string &cfg_path);
  ~TargetQuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override; // Not used
  // Scalar targetStep(const polynomial_trajectories::PolynomialTrajectory &trajectory_, Ref<Vector<>> obs);
  Scalar targetStep(Ref<Vector<>> obs);

  //
  Vector<3> getPosition(void) const;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os, const TargetQuadrotorEnv &quad_env);

 private:
  // Quadrotor
  std::shared_ptr<TargetQuadrotor> target_ptr_;
  QuadState quad_state_;
  Logger logger_{"TargetQaudrotorEnv"};

  // // Minimum snap trajectory
  MinimumSnapTrajectory trajectory_;
  Scalar sim_time_;

  // Observations and actions (for RL)
  Vector<targetquadenv::kNObs> quad_obs_;

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;

  // Auxiliar variables
  Scalar sim_dt_, max_t_;
};

}  // namespace flightlib
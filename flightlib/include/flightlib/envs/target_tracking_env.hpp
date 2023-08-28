//
// This is inspired by RaiGym, thanks.
//
#pragma once

// std
#include <memory>

// openmp
#include <omp.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/tracker_quadrotor_env.hpp"
#include "flightlib/envs/quadrotor_env/target_quadrotor_env.hpp"



namespace flightlib {

template<typename EnvBase>
class TargetTrackingEnv {
 public:
  // TargetTrackingEnv();
  // TargetTrackingEnv(const YAML::Node& cfgs_node);
  TargetTrackingEnv(const std::string& cfgs, const bool from_file);
  TargetTrackingEnv(const std::string& cfgs, const std::string& cfgs_quadrotor, const bool from_file);
  ~TargetTrackingEnv();

  // - public OpenAI-gym style functions for vectorized environment
  bool reset(Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs);
  bool step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs, Ref<Vector<>> reward,
            Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info);
  void close();

  // public set functions
  void setSeed(const int seed);

  // public get functions
  void getObs(Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs);
  size_t getEpisodeLength(void);

  // - auxiliary functions
  void isTerminalState(Ref<BoolVector<>> terminal_state);

  // flightmare (visualization)
  bool setUnity(bool render);
  bool connectUnity();
  void disconnectUnity();

  // public functions
  inline int getSeed(void) { return seed_; };
  inline SceneID getSceneID(void) { return scene_id_; };
  inline bool getUnityRender(void) { return unity_render_; };
  inline int getObsDim(void) { return obs_dim_; };
  inline int getTargetObsDim(void) { return target_obs_dim_; };
  inline int getActDim(void) { return act_dim_; };
  inline int getExtraInfoDim(void) { return extra_info_names_.size(); };
  inline int getNumOfEnvs(void) { return envs_.size(); };
  inline std::vector<std::string>& getExtraInfoNames() { return extra_info_names_; };

 private:
  // initialization
  void init(void);
  // step every environment
  void targetStep(Ref<Vector<>> target_obs); // for target quadrotor
  void perTrackerStep(int agent_id, Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<Vector<>> reward,
                      Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info); // for multi-tracking quadrotors

  // create objects
  Logger logger_{"TargetTrackingEnv"};
  std::vector<std::unique_ptr<EnvBase>> envs_;
  std::unique_ptr<TargetQuadrotorEnv> target_;
  std::vector<std::string> extra_info_names_;

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};
  RenderMessage_t unity_output_;
  uint16_t receive_id_{0};

  // auxiliar variables
  int seed_, num_envs_, obs_dim_, target_obs_dim_, act_dim_;
  Matrix<> obs_dummy_;

  // yaml configurations
  YAML::Node cfg_, cfg_target_;
};

}  // namespace flightlib

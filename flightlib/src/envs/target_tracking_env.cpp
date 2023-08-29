#include "flightlib/envs/target_tracking_env.hpp"

namespace flightlib
{

// template<typename EnvBase>
// TargetTrackingEnv<EnvBase>::TargetTrackingEnv() : TargetTrackingEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/target_tracking_env.yaml")) {}

// template<typename EnvBase>
// TargetTrackingEnv<EnvBase>::TargetTrackingEnv(const YAML::Node& cfg_node) : cfg_(cfg_node)
// {
//   // Initialization
//   init();
// }

template<typename EnvBase>
TargetTrackingEnv<EnvBase>::TargetTrackingEnv(const std::string& cfgs, const bool from_file)
{
  // Load environment configuration
  if (from_file)
  {
    // Load directly from a yaml file
    cfg_ = YAML::LoadFile(cfgs);
  }
  else
  {
    // Load from a string or dictionary
    cfg_ = YAML::Load(cfgs);
  }
  // Initialization
  init();
}

template<typename EnvBase>
TargetTrackingEnv<EnvBase>::TargetTrackingEnv(const std::string& cfgs, const std::string& cfgs_target, const bool from_file)
{
  // Load environment configuration
  if (from_file)
  {
    // Load directly from a yaml file
    cfg_ = YAML::LoadFile(cfgs);
    cfg_target_ = YAML::LoadFile(cfgs_target);
  }
  else
  {
    // Load from a string or dictionary
    cfg_ = YAML::Load(cfgs);
    cfg_target_ = YAML::Load(cfgs_target);
  }
  // Initialization
  init();
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::init(void)
{
  std::cout << "Initialize Flightmare Target Tracking Environment" << std::endl;

  unity_render_ = cfg_["env"]["render"].as<bool>();
  seed_ = cfg_["env"]["seed"].as<int>();
  num_envs_ = cfg_["env"]["num_envs"].as<int>();
  scene_id_ = cfg_["env"]["scene_id"].as<SceneID>();

  // Set threads
  omp_set_num_threads(cfg_["env"]["num_threads"].as<int>());

  // Create & setup environments (environment means quadrotor)
  const bool render = false;
  for (int i = 0; i < num_envs_; i++)
  {
    envs_.push_back(std::make_unique<EnvBase>());
  }

  // Define target drone
  target_ = std::make_unique<TargetQuadrotorEnv>();

  // Set Unity
  setUnity(unity_render_);

  obs_dim_ = envs_[0]->getObsDim();
  target_obs_dim_ = target_->getObsDim();
  act_dim_ = envs_[0]->getActDim();

  // Generate reward names
  // Compute it once to get reward names. actual value is not used
  envs_[0]->updateExtraInfo();
  for (auto& re : envs_[0]->extra_info_)
  {
    extra_info_names_.push_back(re.first);
  }
}

template<typename EnvBase>
TargetTrackingEnv<EnvBase>::~TargetTrackingEnv() {}

template<typename EnvBase>
bool TargetTrackingEnv<EnvBase>::reset(Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs)
{
  if (obs.rows() != num_envs_ || obs.cols() != obs_dim_) 
  {
    logger_.error("Input matrix dimensions do not match with that of the environment.");
    return false;
  }

  receive_id_ = 0;
  for (int i = 0; i < num_envs_; i++)
  {
    envs_[i]->reset(obs.row(i));
  }

  // Initialize target quadrotor
  target_->reset(target_obs);

  return true;
}

template<typename EnvBase>
bool TargetTrackingEnv<EnvBase>::step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs, Ref<Vector<>> reward,
                                      Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info)
{
    if (act.rows() != num_envs_ || act.cols() != act_dim_ || obs.rows() != num_envs_ || obs.cols() != obs_dim_ || reward.rows() != num_envs_ ||
        reward.cols() != 1 || done.rows() != num_envs_ || done.cols() != 1 || extra_info.rows() != num_envs_ || extra_info.cols() != extra_info_names_.size())
    {
      logger_.error("Input matrix dimensions do not match with that of the environment.");
      return false;
    }

#pragma omp parallel for schedule(dynamic)
  // For tracker quadrotor
  for (int i = 0; i < num_envs_; i++)
  {
    perTrackerStep(i, act, obs, reward, done, extra_info);
  }

  // For target quadrotor
  targetStep(target_obs);

  if (unity_render_ && unity_ready_)
  {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();
  }
  return true;
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::close()
{
  for (int i = 0; i < num_envs_; i++)
  {
    envs_[i]->close();
  }
  target_->close();
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::setSeed(const int seed)
{
  int seed_inc = seed;
  for (int i = 0; i < num_envs_; i++) envs_[i]->setSeed(seed_inc++);
  target_->setSeed(seed_inc++);
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::getObs(Ref<MatrixRowMajor<>> obs, Ref<Vector<>> target_obs)
{
  for (int i = 0; i < num_envs_; i++) envs_[i]->getObs(obs.row(i));
  target_->getObs(target_obs);
}

template<typename EnvBase>
size_t TargetTrackingEnv<EnvBase>::getEpisodeLength(void)
{
  if (envs_.size() <= 0)
  {
    return 0;
  }
  else
  {
    return (size_t)envs_[0]->getMaxT() / envs_[0]->getSimTimeStep();
  }
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::targetStep(Ref<Vector<>> target_obs)
{
  // Target reward is useless
  // target_->targetStep(trajectory_, target_obs);
  target_->targetStep(target_obs);
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::perTrackerStep(int agent_id, Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<Vector<>> reward,
                                                Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info)
{
  reward(agent_id) = envs_[agent_id]->trackerStep(act.row(agent_id), obs.row(agent_id), target_->getPosition());

  Scalar terminal_reward = 0;
  done(agent_id) = envs_[agent_id]->isTerminalState(terminal_reward);

  envs_[agent_id]->updateExtraInfo();
  for (int j = 0; j < extra_info.cols(); j++)
  {
    extra_info(agent_id, j) = envs_[agent_id]->extra_info_[extra_info_names_[j]];
  }

  if (done[agent_id])
  {
    envs_[agent_id]->reset(obs.row(agent_id));
    reward(agent_id) += terminal_reward;
  }
}

template<typename EnvBase>
bool TargetTrackingEnv<EnvBase>::setUnity(bool render)
{
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr)
  {
    // Create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // Add objects (tracking quadrotors) to Unity
    for (int i = 0; i < num_envs_; i++)
    {
      envs_[i]->addObjectsToUnity(unity_bridge_ptr_);
    }
    // // Add target quadrotor to Unity
    target_->addObjectsToUnity(unity_bridge_ptr_);

    logger_.info("Flightmare Bridge is created.");
  }
  return true;
}

template<typename EnvBase>
bool TargetTrackingEnv<EnvBase>::connectUnity(void)
{
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::isTerminalState(Ref<BoolVector<>> terminal_state) {}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::disconnectUnity(void)
{
  if (unity_bridge_ptr_ != nullptr)
  {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  }
  else
  {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

// IMPORTANT. Otherwise:
// Segmentation fault (core dumped)
template class TargetTrackingEnv<TrackerQuadrotorEnv>;

}  // namespace flightlib

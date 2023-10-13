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
void TargetTrackingEnv<EnvBase>::init(void)
{
  std::cout << "Initialize Flightmare Target Tracking Environment" << std::endl;

  unity_render_ = cfg_["env"]["render"].as<bool>();
  seed_ = cfg_["env"]["seed"].as<int>();
  num_envs_ = cfg_["env"]["num_envs"].as<int>();
  num_targets_ = cfg_["env"]["num_targets"].as<int>();
  scene_id_ = cfg_["env"]["scene_id"].as<SceneID>();

  // Set threads
  omp_set_num_threads(cfg_["env"]["num_threads"].as<int>());

  // Create & setup environments (environment means quadrotor)
  const bool render = false;
  for (int i = 0; i < num_envs_; i++)
  {
    envs_.push_back(std::make_unique<EnvBase>());
  }

  // Define target drones
  for (int i = 0; i < num_targets_; i++)
  {
    targets_.push_back(std::make_unique<TargetQuadrotorEnv>());
  }

  // Set initial start position
  target_positions_.push_back(Vector<3>{3.0, 3.0, 5.0});
  target_positions_.push_back(Vector<3>{3.0, -3.0, 5.0});
  target_positions_.push_back(Vector<3>{-3.0, -3.0, 5.0});
  target_positions_.push_back(Vector<3>{-3.0, 3.0, 5.0});

  // Target minimum snap trajectory
  Eigen::MatrixXf way_points(7, 3); // Should be n
  way_points << 0, 0, 5,   3, 2, 7,   3, 4, 4,   0, 6, 5,   -3, 4, 7,   -3, 2, 4,   0, 0, 5; // 6m x 6m circle
  Eigen::VectorXf segment_times(6); // Should be n-1
  segment_times << 1.5, 1.5, 1.5, 1.5, 1.5, 1.5;

  MinimumSnapTrajectory trajectory1 = MinimumSnapTrajectory();
  trajectory1.setMinimumSnapTrajectory(way_points, segment_times);

  way_points << 0, 0, 5,   -3, 2, 4,   -3, 4, 7,   0, 6, 5,   3, 4, 4,   3, 2, 7,   0, 0, 5; // 6m x 6m circle
  segment_times << 1.5, 1.5, 1.5, 1.5, 1.5, 1.5;

  MinimumSnapTrajectory trajectory2 = MinimumSnapTrajectory();
  trajectory2.setMinimumSnapTrajectory(way_points, segment_times);

  trajectories_.push_back(trajectory1);
  trajectories_.push_back(trajectory2);
  trajectories_.push_back(trajectory1);
  trajectories_.push_back(trajectory2);

  // Set Unity
  setUnity(unity_render_);

  obs_dim_ = envs_[0]->getObsDim();
  target_obs_dim_ = targets_[0]->getObsDim();
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
bool TargetTrackingEnv<EnvBase>::reset(Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> target_obs)
{
  if (obs.rows() != num_envs_ || obs.cols() != obs_dim_) {
    logger_.error("Input matrix dimensions do not match with that of the environment.");
    return false;
  }

  receive_id_ = 0;

  for (int i = 0; i < num_targets_; i++) {
    targets_[i]->reset(target_obs.row(i), target_positions_[i], trajectories_[i]);
  }

  // Initial target position
  std::vector<Vector<3>> tracker_positions;
  
  // tracker_positions.push_back(Vector<3>{0.0, -8.0, 5.0});
  for (int i = 0; i < num_envs_; i++) {
    tracker_positions.push_back(Vector<3>{uniform_dist_(random_gen_) * 7.0, uniform_dist_(random_gen_) * 7.0, uniform_dist_(random_gen_) + 5});
  }
  tracker_positions_ = tracker_positions;


  for (int i = 0; i < num_envs_; i++) {
    std::vector<Vector<3>> other_tracker_positions;
    for (int j = 0; j < num_envs_; j++) {
      if (i != j)
        other_tracker_positions.push_back(tracker_positions_[j]);
    }
    envs_[i]->reset(obs.row(i), tracker_positions_[i], target_positions_, other_tracker_positions);
  }

  return true;
}

template<typename EnvBase>
bool TargetTrackingEnv<EnvBase>::step(Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> target_obs, Ref<Vector<>> reward,
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
  for (int i = 0; i < num_targets_; i++)
  {
    perTargetStep(i, target_obs.row(i));
  }

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
  for (int i = 0; i < num_envs_; i++)
  {
    targets_[i]->close();
  }
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::setSeed(const int seed)
{
  int seed_inc = seed;
  for (int i = 0; i < num_envs_; i++) envs_[i]->setSeed(seed_inc++);
  for (int i = 0; i < num_targets_; i++) targets_[i]->setSeed(seed_inc++);
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::getObs(Ref<MatrixRowMajor<>> obs, Ref<MatrixRowMajor<>> target_obs)
{
  for (int i = 0; i < num_envs_; i++) envs_[i]->getObs(obs.row(i));
  for (int i = 0; i < num_targets_; i++) targets_[i]->getObs(target_obs.row(i));
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
void TargetTrackingEnv<EnvBase>::perTargetStep(int target_id, Ref<Vector<>> target_obs)
{
  targets_[target_id]->targetStep(target_obs);
}

template<typename EnvBase>
void TargetTrackingEnv<EnvBase>::perTrackerStep(int agent_id, Ref<MatrixRowMajor<>> act, Ref<MatrixRowMajor<>> obs, Ref<Vector<>> reward,
                                                Ref<BoolVector<>> done, Ref<MatrixRowMajor<>> extra_info)
{
  std::vector<Vector<3>> other_tracker_positions;
  for (int i = 0; i < num_envs_; i++) {
    if (i != agent_id)
      other_tracker_positions.push_back(envs_[i]->getPosition());
  }
  std::vector<Vector<3>> target_positions;
  for (int i = 0; i < num_targets_; i++) {
    target_positions.push_back(targets_[i]->getPosition());
  }

  // std::cout << "Tracker " << agent_id << " =========================================" << std::endl;
  reward(agent_id) = envs_[agent_id]->trackerStep(act.row(agent_id), obs.row(agent_id), target_positions, other_tracker_positions);

  // // Compute average minimum target covariance
  // int num_trackers = num_envs_;
  // std::vector<std::vector<Scalar>>;
  // for (int i = 0; i < num_trackers; ++i) {
  //   std::vector<Scalar> target_cov = envs_[agent_id]->getTargetStateErrorCovarianceNorms();
  //   // Store only minimum covariance
  // }


  Scalar terminal_reward = 0;
  done(agent_id) = envs_[agent_id]->isTerminalState(terminal_reward);

  envs_[agent_id]->updateExtraInfo();
  for (int j = 0; j < extra_info.cols(); j++) {
    extra_info(agent_id, j) = envs_[agent_id]->extra_info_[extra_info_names_[j]];
  }

  if (done[agent_id]) {
    envs_[agent_id]->reset(obs.row(agent_id), tracker_positions_[agent_id], target_positions, other_tracker_positions);
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
    // Add objects to Unity
    for (int i = 0; i < num_envs_; i++) envs_[i]->addObjectsToUnity(unity_bridge_ptr_);
    for (int i = 0; i < num_targets_; i++) targets_[i]->addObjectsToUnity(unity_bridge_ptr_);

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

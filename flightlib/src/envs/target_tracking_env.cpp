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


  // // **************************** trajectory 2 ****************************
  // target_positions_.push_back(Vector<3>{0.0, 8.0, 5.0});
  // target_positions_.push_back(Vector<3>{0.0, -8.0, 5.0});
  // target_positions_.push_back(Vector<3>{8.0, 0.0, 5.0});
  // target_positions_.push_back(Vector<3>{-8.0, 0.0, 5.0});

  // // Target minimum snap trajectory
  // Eigen::MatrixXf way_points(5, 3); // Should be n
  // Eigen::VectorXf segment_times(4); // Should be n-1

  // way_points << 0, 8, 5,   8, 0, 5,   0, -8, 5,   -8, 0, 5,   0, 8, 5; // 6m x 6m circle
  // segment_times << 3.0, 3.0, 3.0, 3.0;
  // MinimumSnapTrajectory trajectory1 = MinimumSnapTrajectory();
  // trajectory1.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 0, -8, 5,   -8, 0, 5,   0, 8, 5,   8, 0, 5,   0, -8, 5;
  // segment_times << 3.0, 3.0, 3.0, 3.0;
  // MinimumSnapTrajectory trajectory2 = MinimumSnapTrajectory();
  // trajectory2.setMinimumSnapTrajectory(way_points, segment_times);

  // // way_points << 8, 0, 5,   0, -8, 5,   -8, 0, 5,   0, 8, 5,   8, 0, 5;
  // // segment_times << 3.0, 3.0, 3.0, 3.0;
  // // MinimumSnapTrajectory trajectory3 = MinimumSnapTrajectory();
  // // trajectory3.setMinimumSnapTrajectory(way_points, segment_times);

  // // way_points << -8, 0, 5,   0, 8, 5,   8, 0, 5,   0, -8, 5,   -8, 0, 5;
  // // segment_times << 3.0, 3.0, 3.0, 3.0;
  // // MinimumSnapTrajectory trajectory4 = MinimumSnapTrajectory();
  // // trajectory4.setMinimumSnapTrajectory(way_points, segment_times);

  // trajectories_.push_back(trajectory1);
  // trajectories_.push_back(trajectory2);
  // // trajectories_.push_back(trajectory3);
  // // trajectories_.push_back(trajectory4);


  // **************************** trajectory 3 ****************************
  target_positions_.push_back(Vector<3>{-4.0, 8.0, 5.0});
  target_positions_.push_back(Vector<3>{4.0, -8.0, 5.0});
  // target_positions_.push_back(Vector<3>{8.0, 4.0, 8.0});
  // target_positions_.push_back(Vector<3>{-8.0, -4.0, 8.0});

  // Target minimum snap trajectory
  Eigen::MatrixXf way_points(5, 3); // Should be n
  Eigen::VectorXf segment_times(4); // Should be n-1

  way_points << -4, 8, 5,   -8, 0, 5,   -4, -8, 5,   -1, 0, 5,   -4, 8, 5; // 6m x 6m circle
  segment_times << 3.5, 3.5, 3.5, 3.5;
  MinimumSnapTrajectory trajectory1 = MinimumSnapTrajectory();
  trajectory1.setMinimumSnapTrajectory(way_points, segment_times);

  way_points << 4, -8, 5,   8, 0, 5,   4, 8, 5,   1, 0, 5,   4, -8, 5;
  segment_times << 3.5, 3.5, 3.5, 3.5;
  MinimumSnapTrajectory trajectory2 = MinimumSnapTrajectory();
  trajectory2.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 8, 4, 8,   0, 8, 8,   -8, 4, 8,   0, 1, 8,   8, 4, 8; // 6m x 6m circle
  // segment_times << 3.5, 3.5, 3.5, 3.5;
  // MinimumSnapTrajectory trajectory3 = MinimumSnapTrajectory();
  // trajectory3.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << -8, -4, 8,   0, -8, 8,   8, -4, 8,   0, -1, 8,   -8, -4, 8;
  // segment_times << 3.5, 3.5, 3.5, 3.5;
  // MinimumSnapTrajectory trajectory4 = MinimumSnapTrajectory();
  // trajectory4.setMinimumSnapTrajectory(way_points, segment_times);

  trajectories_.push_back(trajectory1);
  trajectories_.push_back(trajectory2);
  // trajectories_.push_back(trajectory3);
  // trajectories_.push_back(trajectory4);


  // // **************************** trajectory 1 ****************************

  // // Set initial start position
  // target_positions_.push_back(Vector<3>{-5.0, 8.0, 5.0}); // LU
  // target_positions_.push_back(Vector<3>{5.0, 2.0, 5.0}); // RU
  // target_positions_.push_back(Vector<3>{-5.0, -2.0, 5.0}); // LD
  // target_positions_.push_back(Vector<3>{5.0, -8.0, 5.0}); // RD

  // // Target minimum snap trajectory
  // Eigen::MatrixXf way_points(5, 3); // Should be n
  // Eigen::VectorXf segment_times(4); // Should be n-1

  // way_points << -5, 8, 5,   -8, 5, 5,   -5, 2, 5,   -2, 5, 5,   -5, 8, 5; // 6m x 6m circle
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory1 = MinimumSnapTrajectory();
  // trajectory1.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 5, 2, 5,   2, 5, 5,   5, 8, 5,   8, 5, 5,   5, 2, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory2 = MinimumSnapTrajectory();
  // trajectory2.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << -5, -2, 5,   -2, -5, 5,   -5, -8, 5,   -8, -5, 5,   -5, -2, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory3 = MinimumSnapTrajectory();
  // trajectory3.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 5, -8, 5,   8, -5, 5,   5, -2, 5,   2, -5, 5,   5, -8, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory4 = MinimumSnapTrajectory();
  // trajectory4.setMinimumSnapTrajectory(way_points, segment_times);

  // trajectories_.push_back(trajectory1);
  // trajectories_.push_back(trajectory2);
  // trajectories_.push_back(trajectory3);
  // trajectories_.push_back(trajectory4);


  // // **************************** trajectory new for 6 targets ****************************

  // // Set initial start position
  // target_positions_.push_back(Vector<3>{-5.0, 8.0, 5.0}); // LU
  // target_positions_.push_back(Vector<3>{5.0, 2.0, 5.0}); // RU
  // target_positions_.push_back(Vector<3>{-5.0, -2.0, 5.0}); // LD
  // target_positions_.push_back(Vector<3>{5.0, -8.0, 5.0}); // RD
  // target_positions_.push_back(Vector<3>{0.0, 8.0, 8.0});
  // target_positions_.push_back(Vector<3>{0.0, -8.0, 8.0});

  // // Target minimum snap trajectory
  // Eigen::MatrixXf way_points(5, 3); // Should be n
  // Eigen::VectorXf segment_times(4); // Should be n-1

  // way_points << -5, 8, 5,   -8, 5, 5,   -5, 2, 5,   -2, 5, 5,   -5, 8, 5; // 6m x 6m circle
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory1 = MinimumSnapTrajectory();
  // trajectory1.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 5, 2, 5,   2, 5, 5,   5, 8, 5,   8, 5, 5,   5, 2, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory2 = MinimumSnapTrajectory();
  // trajectory2.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << -5, -2, 5,   -2, -5, 5,   -5, -8, 5,   -8, -5, 5,   -5, -2, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory3 = MinimumSnapTrajectory();
  // trajectory3.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 5, -8, 5,   8, -5, 5,   5, -2, 5,   2, -5, 5,   5, -8, 5;
  // segment_times << 2.0, 2.0, 2.0, 2.0;
  // MinimumSnapTrajectory trajectory4 = MinimumSnapTrajectory();
  // trajectory4.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 0, 8, 8,   8, 0, 8,   0, -8, 8,   -8, 0, 8,   0, 8, 8; // 6m x 6m circle
  // segment_times << 3.0, 3.0, 3.0, 3.0;
  // MinimumSnapTrajectory trajectory5 = MinimumSnapTrajectory();
  // trajectory5.setMinimumSnapTrajectory(way_points, segment_times);

  // way_points << 0, -8, 8,   -8, 0, 8,   0, 8, 8,   8, 0, 8,   0, -8, 8;
  // segment_times << 3.0, 3.0, 3.0, 3.0;
  // MinimumSnapTrajectory trajectory6 = MinimumSnapTrajectory();
  // trajectory6.setMinimumSnapTrajectory(way_points, segment_times);

  // trajectories_.push_back(trajectory1);
  // trajectories_.push_back(trajectory2);
  // trajectories_.push_back(trajectory3);
  // trajectories_.push_back(trajectory4);
  // trajectories_.push_back(trajectory5);
  // trajectories_.push_back(trajectory6);

  // // **********************************************************************


  // Data recoder
  multi_save_ = std::make_shared<MultiAgentSave>();
  multi_save_->init(num_targets_);

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

  // // Ideal multi position
  // tracker_positions.push_back(Vector<3>{0.0, 15.0, 5.0});
  // tracker_positions.push_back(Vector<3>{-11.0, -11.0, 5.0});
  // tracker_positions.push_back(Vector<3>{11.0, -11.0, 5.0});

  // // Unideal multi position
  // tracker_positions.push_back(Vector<3>{3.0, -15.0, 5.0});
  // tracker_positions.push_back(Vector<3>{0.0, -15.0, 5.0});
  // tracker_positions.push_back(Vector<3>{-3.0, -15.0, 5.0});


  // // Training multi position
  // tracker_positions.push_back(Vector<3>{0.0, 13.0, 5.0});
  // tracker_positions.push_back(Vector<3>{-10.0, -10.0, 5.0});
  // tracker_positions.push_back(Vector<3>{10.0, -10.0, 5.0});

  for (int i = 0; i < num_envs_; i++) {
    Scalar theta = uniform_theta_(random_gen_) * M_PI;
    Scalar radius = 13.0;
    // Scalar radius = uniform_radius_(random_gen_);
    Scalar random_x = radius * cos(theta);
    Scalar random_y = radius * sin(theta);
    Scalar random_z = uniform_altitude_(random_gen_);
    // Scalar random_z = 10.0;
    tracker_positions.push_back(Vector<3>{random_x, random_y, random_z});
  }
  tracker_positions_ = tracker_positions;


  for (int i = 0; i < num_envs_; i++) {
    std::vector<Vector<3>> other_tracker_positions;
    for (int j = 0; j < num_envs_; j++) {
      if (i != j)
        other_tracker_positions.push_back(tracker_positions_[j]);
    }
    envs_[i]->reset(obs.row(i), tracker_positions_[i], target_positions_, other_tracker_positions, i);
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
  // For tracker quadrotor (in single agent setting)
  for (int i = 0; i < num_envs_; i++)
  {
    perTrackerStep(i, act, obs, reward, done, extra_info);
  }



  //************************************************************************
  //*************************** Global Reward ******************************
  //************************************************************************

  // Scalar w = 2.0;
  // Scalar individual_reward = 0.0; // Max individual reward 1.3 * 3 = 3.9
  // Scalar cooperative_reward = computeGlobalReward(); // Max cooperative reward = w * 1.0

  // for (int i = 0; i < num_envs_; i++)
  // {
  //   individual_reward += reward(i);
  // }

  // Scalar global_reward = individual_reward + w * cooperative_reward;

  // for (int i = 0; i < num_envs_; i++)
  // {
  //   reward(i) = global_reward;
  //   // reward(i) = global_reward / 10; // Reward scaling
  // }


  // //************************************************************************
  // // For seperated network
  // Scalar w = 0.3;
  // Scalar cooperative_reward = computeGlobalReward();

  // for (int i = 0; i < num_envs_; i++)
  // {
  //   reward(i) += w * cooperative_reward; // individual reward + w * cooperative reward, (1.3 + w * 1.0)
  // }


  // //************************************************************************
  // // // For pseudo single network
  // Scalar w = 0.3;
  // Scalar cooperative_reward = computeGlobalReward();
  // reward(0) += w * cooperative_reward; // individual reward + w * cooperative reward, (1.3 + w * 1.0)


  // //************************************************************************
  // // For multi control policy network
  // Scalar global_reward = 0.0;
  // for (int i = 0; i < num_envs_; i++)
  // {
  //   global_reward += reward(i);
  // }
  // global_reward /= num_envs_;

  // for (int i = 0; i < num_envs_; i++)
  // {
  //   reward(i) = global_reward;
  // }

  //************************************************************************
  //*************************** Global Reward ******************************
  //************************************************************************




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

  // Get 0 reward when multi-agent setting
  reward(agent_id) = envs_[agent_id]->trackerStep(act.row(agent_id), obs.row(agent_id), target_positions, other_tracker_positions);

  Scalar terminal_reward = 0;
  done(agent_id) = envs_[agent_id]->isTerminalState(terminal_reward);

  envs_[agent_id]->updateExtraInfo();
  for (int j = 0; j < extra_info.cols(); j++) {
    extra_info(agent_id, j) = envs_[agent_id]->extra_info_[extra_info_names_[j]];
  }

  if (done[agent_id]) {
    envs_[agent_id]->reset(obs.row(agent_id), tracker_positions_[agent_id], target_positions, other_tracker_positions, agent_id);
    reward(agent_id) += terminal_reward;
  }
}

template<typename EnvBase>
Scalar TargetTrackingEnv<EnvBase>::computeGlobalReward() {

  // // 3 targets
  // std::vector<Scalar> min_cov_det_list = {std::numeric_limits<Scalar>::infinity(),
  //                                         std::numeric_limits<Scalar>::infinity(),
  //                                         std::numeric_limits<Scalar>::infinity()};
  // std::vector<Vector<3>> min_position = {Vector<3>(), Vector<3>(), Vector<3>()};
  // std::vector<Matrix<3, 3>> min_cov = {Matrix<3, 3>(), Matrix<3, 3>(), Matrix<3, 3>()};


  // 4 targets
  std::vector<Scalar> min_cov_det_list = {std::numeric_limits<Scalar>::infinity(),
                                          std::numeric_limits<Scalar>::infinity(),
                                          std::numeric_limits<Scalar>::infinity(),
                                          std::numeric_limits<Scalar>::infinity()};
  std::vector<Vector<3>> min_position = {Vector<3>(), Vector<3>(), Vector<3>(), Vector<3>()};
  std::vector<Matrix<3, 3>> min_cov = {Matrix<3, 3>(), Matrix<3, 3>(), Matrix<3, 3>(), Matrix<3, 3>()};


  // Hungarian algorithm
  std::vector<std::vector<Scalar>> cost_matrix;

  for (int i = 0; i < num_envs_; ++i) {
    // Initialize cost matrix
    for (int j = 0; j < num_targets_; ++j) {
      std::vector<Scalar> place_holder(num_targets_, 1000.0);
      cost_matrix.push_back(place_holder);
    }    
    
    // Compute cost
    for (int j = 0; j < num_targets_; ++j) {
      // agent i의 대상 k에 대한 추정치 - target j gt
      for (int k = 0; k < num_targets_; ++k) {
        Scalar cost = (envs_[i]->getEstimatedTargetPosition(k) - targets_[j]->getPosition()).norm();
        cost_matrix[j][k] = cost;
      }
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(cost_matrix, assignment);

    for (int j = 0; j < num_targets_; ++j) {
      // gt target j에 matching된 추정 대상
      Scalar cov_det = envs_[i]->getTargetPositionCovDet(assignment[j]);
      if (cov_det < min_cov_det_list[j]) {
        min_cov_det_list[j] = cov_det;
        min_position[j] = envs_[i]->getEstimatedTargetPosition(assignment[j]);
        min_cov[j] = envs_[i]->getTargetPositionCov(assignment[j]);
      }
    }
  }

  Scalar avg_3_sigma = 0.0;
  for (int i = 0; i < num_targets_; ++i) {
    Scalar min_cov_det = min_cov_det_list[i];
    Scalar clipped_det = std::max(static_cast<Scalar>(0.0), min_cov_det); // Det must be zero or positive
    avg_3_sigma += 27 * sqrt(clipped_det);
  }
  avg_3_sigma /= num_targets_;

  // Scalar cooperative_reward = exp(-0.1 * pow(avg_3_sigma, 5)); // Original reward
  Scalar cooperative_reward = exp(-20.0 * pow(avg_3_sigma, 5));


  if (std::isnan(cooperative_reward)) {
    std::cout << "nan occurs from cooperative_reward" << std::endl;
    std::cout << "avg_3_sigma  : " << avg_3_sigma << std::endl;
    exit(0);
  }

  // std::cout << "-----------------------------------------" << std::endl;
  // std::cout << "min cov det        : " << min_cov_det_list[0] << ", " << min_cov_det_list[1] << ", " << min_cov_det_list[2] << ", " << min_cov_det_list[3] << std::endl;
  // std::cout << "avg cov det        : " << avg_3_sigma << std::endl;
  // std::cout << "cooperative reward : " << cooperative_reward << std::endl;
  // std::cout << "-----------------------------------------" << std::endl;



  //************************************************************************
  //*************************** Data Recoder *******************************
  //************************************************************************

  if (num_envs_ > 1) {
    if (!multi_save_->isFull()) {
      multi_save_->store(min_position, min_cov, 0.02);
    }
    else if (multi_flag_ && multi_save_->isFull()) {
      multi_save_->save();
      multi_flag_ = false;
      std::cout << ">>> Multi-Agent output save is done" << std::endl;
    }
  }

  //************************************************************************
  //*************************** Data Recoder *******************************
  //************************************************************************


  return cooperative_reward;
}

template<typename EnvBase>
std::pair<Vector<3>, Matrix<3, 3>> TargetTrackingEnv<EnvBase>::fuseGaussian(const Ref<Vector<3>> mu1, const Ref<Matrix<3, 3>> cov1,
                                                                            const Ref<Vector<3>> mu2, const Ref<Matrix<3, 3>> cov2) {
  // Calculate the information matrices
  Matrix<3, 3> Y1 = cov1.inverse();
  Matrix<3, 3> Y2 = cov2.inverse();
  
  // Calculate the information vectors
  Vector<3> eta1 = Y1 * mu1;
  Vector<3> eta2 = Y2 * mu2;
  
  // Combine the information
  Matrix<3, 3> Y = Y1 + Y2;
  Vector<3> eta = eta1 + eta2;
  
  // Convert back to mean and covariance
  Matrix<3, 3> fused_cov = Y.inverse();
  Vector<3> fused_mu = fused_cov * eta;
  
  return std::make_pair(fused_mu, fused_cov);
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

#include "flightlib/envs/quadrotor_env/tracker_quadrotor_env.hpp"

namespace flightlib {

TrackerQuadrotorEnv::TrackerQuadrotorEnv() : TrackerQuadrotorEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/tracker_quadrotor_env.yaml")) {}
TrackerQuadrotorEnv::TrackerQuadrotorEnv(const std::string &cfg_path) : EnvBase() {

  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  // load parameters
  loadParam(cfg_);

  //
  tracker_ptr_ = std::make_shared<TrackerQuadrotor>();

  //
  num_cameras_ = 1;

  // Mount front, left, right stereo cameras
  for (int i = 0; i < num_cameras_; i++) {
    multi_stereo_.push_back(std::make_shared<StereoCamera>());
  }

  d_l_ = Vector<3>(0.06, 0.0, -0.1); // body origin w.r.t. left camera
  d_r_ = Vector<3>(-0.06, 0.0, -0.1); // body origin w.r.t. right camera
  R_B_C_ = (Rot_x(-M_PI_2) * Rot_y(M_PI_2)).inverse(); // from body to camera

  // Frame translation matrix
  T_B_LC_.block<3, 3>(0, 0) = R_B_C_;
  T_B_LC_.block<3, 1>(0, 3) = d_l_;
  T_B_LC_.row(3) << 0.0, 0.0, 0.0, 1.0;

  // Transformation from left camera to body
  T_LC_B_ = T_B_LC_.inverse();

  // Front camera
  multi_stereo_[0]->init(d_l_, d_r_, R_B_C_);

  // // Data recoder
  sensor_save_ = std::make_shared<SensorSave>();
  tracking_save_ = std::make_shared<TrackingSave>();

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  tracker_ptr_->updateDynamics(dynamics);
  tracker_ptr_->setVelocityPIDGain(kp_vxy_, ki_vxy_, kd_vxy_, kp_vz_, ki_vz_, kd_vz_, kp_angle_, ki_angle_, kd_angle_, kp_wz_, ki_wz_, kd_wz_);

  // define a bounding box
  world_box_ << -50, 50, -50, 50, 0, 50;
  if (!tracker_ptr_->setWorldBox(world_box_))
  {
    logger_.error("cannot set wolrd box");
  };



  // define input and output dimension for the environment
  // obs_dim_ = 28; // Three targets & ego
  
  // obs_dim_ = 37; // single (2 targets)
  // obs_dim_ = 55; // single (4 targets)

  obs_dim_ = 56; // multi (2 agents & 3 targets)
  // obs_dim_ = 73; // multi (3 agents & 4 targets)


  act_dim_ = trackerquadenv::kNAct;
}

TrackerQuadrotorEnv::~TrackerQuadrotorEnv() {}

bool TrackerQuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (false) {
    // randomly reset the quadrotor state
    // reset position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) + 5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  else
  {
    quad_state_.x(QS::POSX) = 0.0;
    quad_state_.x(QS::POSY) = -8.0;
    quad_state_.x(QS::POSZ) = 5.0;
  }
  // Reset quadrotor with random states
  tracker_ptr_->reset(quad_state_);
  // Reset velocity control command
  cmd_.t = 0.0;
  cmd_.velocity.setZero();

  // cmd_.collective_thrust = 0.0;
  // cmd_.omega.setZero();

  // cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool TrackerQuadrotorEnv::reset(Ref<Vector<>> obs, Ref<Vector<>> position,
                                const std::vector<Vector<3>>& target_positions, const std::vector<Vector<3>>& tracker_positions, const int agent_id) {
  agent_id_ = agent_id;
  
  quad_state_.setZero();
  quad_act_.setZero();

  quad_state_.x(QS::POSX) = position[0];
  quad_state_.x(QS::POSY) = position[1];
  quad_state_.x(QS::POSZ) = position[2];

  // Scalar yaw = uniform_dist_(random_gen_) * M_PI;
  // Scalar yaw = M_PI_2;
  Scalar yaw = atan2(-position[1], -position[0]); // Center oriented yaw angle
  
  Vector<3> euler(yaw, 0, 0);
  Vector<4> quaternion = eulerToQuaternion(euler);

  quad_state_.x(QS::ATTW) = quaternion[0];
  quad_state_.x(QS::ATTX) = quaternion[1];
  quad_state_.x(QS::ATTY) = quaternion[2];
  quad_state_.x(QS::ATTZ) = quaternion[3];

  // Reset quadrotor with random states
  tracker_ptr_->reset(quad_state_);

  //
  num_targets_ = target_positions.size();
  num_trackers_ = tracker_positions.size();
  // num_trackers_ = 0; // When training single agent ppo

  // Initialize multi kalman filter for target
  for (int i = 0; i < num_targets_; ++i){
    std::shared_ptr<KalmanFilter> target_kf = std::make_shared<KalmanFilter>();
    Vector<3> target = target_positions[i];
    Vector<6> x0 = (Vector<6>() << target[0], 0, target[1], 0, target[2], 0).finished();
    target_kf->init(sim_dt_, x0);
    target_kalman_filters_.push_back(target_kf);
  }

  // Initialize multi kalman filter for tracker
  for (int i = 0; i < num_trackers_; ++i){
    std::shared_ptr<KalmanFilter> tracker_kf = std::make_shared<KalmanFilter>();
    Vector<3> tracker = tracker_positions[i];
    Vector<6> x0 = (Vector<6>() << tracker[0], 0, tracker[1], 0, tracker[2], 0).finished();
    tracker_kf->init(sim_dt_, x0);
    tracker_kalman_filters_.push_back(tracker_kf);
  }

  // Initialize tracking recoder
  tracking_save_->init(num_targets_, num_trackers_, agent_id_);

  // Reset velocity control command
  cmd_.t = 0.0;
  cmd_.velocity.setZero();

  // cmd_.collective_thrust = 0.0;
  // cmd_.omega.setZero();

  // cmd_.thrusts.setZero();

  // Store ground truth
  gt_target_positions_ = target_positions;
  gt_tracker_positions_ = tracker_positions;

  // obtain observations
  getObs(obs);
  return true;
}

Matrix<3, 3> TrackerQuadrotorEnv::Rot_x(const Scalar angle) const {
  Matrix<3, 3> R_x = (Matrix<3, 3>() << 1, 0, 0,
                                        0, cos(angle), -sin(angle),
                                        0, sin(angle), cos(angle)).finished();
  return R_x;
}

Matrix<3, 3> TrackerQuadrotorEnv::Rot_y(const Scalar angle) const {
  Matrix<3, 3> R_y = (Matrix<3, 3>() << cos(angle), 0, sin(angle),
                                        0, 1, 0,
                                        -sin(angle), 0, cos(angle)).finished();
  return R_y;
}

Matrix<3, 3> TrackerQuadrotorEnv::Rot_z(const Scalar angle) const {
  Matrix<3, 3> R_z = (Matrix<3, 3>() << cos(angle), -sin(angle), 0,
                                        sin(angle), cos(angle), 0,
                                        0, 0, 1).finished();
  return R_z;
}

Matrix<3, 3> TrackerQuadrotorEnv::eulerToRotation(const Ref<Vector<3>> euler_zyx) const {
  Matrix<3, 3> R_x = (Matrix<3, 3>() << 1, 0, 0,
                                        0, cos(euler_zyx[2]), -sin(euler_zyx[2]),
                                        0, sin(euler_zyx[2]), cos(euler_zyx[2])).finished();

  Matrix<3, 3> R_y = (Matrix<3, 3>() << cos(euler_zyx[1]), 0, sin(euler_zyx[1]),
                                        0, 1, 0,
                                        -sin(euler_zyx[1]), 0, cos(euler_zyx[1])).finished();

  Matrix<3, 3> R_z = (Matrix<3, 3>() << cos(euler_zyx[0]), -sin(euler_zyx[0]), 0,
                                        sin(euler_zyx[0]), cos(euler_zyx[0]), 0,
                                        0, 0, 1).finished();
  // Combined rotation matrix
  Matrix<3, 3> R = R_z * R_y * R_x;
  return R;
}

Vector<4> TrackerQuadrotorEnv::eulerToQuaternion(const Ref<Vector<3>> euler_zyx) const {
  Scalar cy = cos(euler_zyx[0] * 0.5);
  Scalar sy = sin(euler_zyx[0] * 0.5);
  Scalar cp = cos(euler_zyx[1] * 0.5);
  Scalar sp = sin(euler_zyx[1] * 0.5);
  Scalar cr = cos(euler_zyx[2] * 0.5);
  Scalar sr = sin(euler_zyx[2] * 0.5);

  Vector<4> quaternion(cy * cp * cr + sy * sp * sr,
                       cy * cp * sr - sy * sp * cr,
                       sy * cp * sr + cy * sp * cr,
                       sy * cp * cr - cy * sp * sr);
  return quaternion;
}

Matrix<4, 4> TrackerQuadrotorEnv::getBodyToWorld() const {
  Matrix<4, 4> T_B_W;
  T_B_W.block<3, 3>(0, 0) = quad_state_.q().toRotationMatrix();
  T_B_W.block<3, 1>(0, 3) = quad_state_.p;
  T_B_W.row(3) << 0.0, 0.0, 0.0, 1.0;

  return T_B_W;
}

Vector<3> TrackerQuadrotorEnv::fromWorldToCamera(Vector<3> point_w) const {
  // Frame transform from world to body
  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_W_B = T_B_W.inverse();

  // Frame transform from world to left camera
  Matrix<3, 4> T_W_LC = (T_B_LC_ * T_W_B).topRows<3>();

  // Covert to homogeneous coordinates
  Vector<4> P_W(point_w[0], point_w[1], point_w[2], 1);

  // From world to left camera
  Vector<3> point_c = T_W_LC * P_W;

  return point_c;
}

Vector<3> TrackerQuadrotorEnv::fromCameraToWorld(Vector<3> point_c) const {
  // Frame transform from world to body
  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_W_B = T_B_W.inverse();

  // Frame transform from world to left camera
  Matrix<4, 4> T_W_LC = (T_B_LC_ * T_W_B);
  // Frame transform from camera to left world
  Matrix<3, 4> T_LC_W = T_W_LC.inverse().topRows<3>();

  // Covert to homogeneous coordinates
  Vector<4> P_C(point_c[0], point_c[1], point_c[2], 1);

  // From world to left camera
  Vector<3> point_w = T_LC_W * P_C;

  return point_w;
}

Matrix<3, 3> TrackerQuadrotorEnv::fromCameraToWorld(Matrix<3, 3> cov) const {
  // Frame rotation from body to world
  Matrix<3, 3> R_B_W = quad_state_.q().toRotationMatrix();

  // Frame rotation from left camera to body
  Matrix<3, 3> R_LC_B = R_B_C_.inverse();

  // Frame rotation from left camera to world
  Matrix<3, 3> R_LC_W = R_B_W * R_LC_B;

  // Transform the 3D error covariance matrix from the camera frame to the world frame
  Matrix<3, 3> cov_w = R_LC_W * cov * R_LC_W.transpose();

  return cov_w;
}

Matrix<3, 3> TrackerQuadrotorEnv::rotateFromCameraToWorld() const {
  // Frame rotation from body to world
  Matrix<3, 3> R_B_W = quad_state_.q().toRotationMatrix();

  // Frame rotation from left camera to body
  Matrix<3, 3> R_LC_B = R_B_C_.inverse();

  // Frame rotation from left camera to world
  Matrix<3, 3> R_LC_W = R_B_W * R_LC_B;

  return R_LC_W;
}

Vector<3> TrackerQuadrotorEnv::getPosition() const {
  return quad_state_.p;
}

Scalar TrackerQuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs)
{
  // Reward function of tracker quadrotor
  Scalar total_reward = 0.0;
  return total_reward;
}

Scalar TrackerQuadrotorEnv::trackerStep(const Ref<Vector<>> act, Ref<Vector<>> obs,
                                        const std::vector<Vector<3>>& target_positions, const std::vector<Vector<3>>& tracker_positions) {
  quad_act_ = act;
  cmd_.t += sim_dt_;
  // cmd_.velocity = quad_act_;

  cmd_.velocity = Vector<4>(quad_act_[0], quad_act_[1], quad_act_[2], quad_act_[3]); // PPO action

  // cmd_.collective_thrust = quad_act_[0] * 22.0;
  // cmd_.omega = Vector<3>(quad_act_[1], quad_act_[2], quad_act_[3]);

  // cmd_.thrusts = Vector<4>(quad_act_[0], quad_act_[1], quad_act_[2], quad_act_[3]) * 5.5;

  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_W_B = T_B_W.inverse(); // World to body

  // Store ground truth position based on world frame
  gt_target_positions_ = target_positions;
  gt_tracker_positions_ = tracker_positions;

  // Try to detect the number of "n" targets
  bool detected = false;
  std::vector<Vector<3>> target_measurements_w;
  std::vector<Vector<3>> target_measurements_c;

  for (int i = 0; i < target_positions.size(); i++) {
    detected = false;
    Vector<3> target_position = target_positions[i];
    for (int j = 0; j < num_cameras_; j++) {
      detected = multi_stereo_[j]->computePixelPoint(target_position, T_W_B);
      if (detected) {
        // Get sensor measurement (based on left camera frame)
        Vector<3> target_measurement_c = multi_stereo_[j]->getSensorMeasurement();
        // Change frame from camera to world
        Vector<3> target_measurement_w = fromCameraToWorld(target_measurement_c);
        target_measurements_w.push_back(target_measurement_w);
        target_measurements_c.push_back(target_measurement_c);
        break;      
      }
    }
  }
  
  // Try to detect the number of "m-1" trackers
  std::vector<Vector<3>> tracker_measurements_w;
  std::vector<Vector<3>> tracker_measurements_c;

  for (int i = 0; i < tracker_positions.size(); i++) {
    detected = false;
    Vector<3> tracker_position = tracker_positions[i];
    for (int j = 0; j < num_cameras_; j++) {
      detected = multi_stereo_[j]->computePixelPoint(tracker_position, T_W_B);
      if (detected) {
        Vector<3> tracker_measurement_c = multi_stereo_[j]->getSensorMeasurement();
        // Change frame from camera to world
        Vector<3> tracker_measurement_w = fromCameraToWorld(tracker_measurement_c);
        tracker_measurements_w.push_back(tracker_measurement_w);
        tracker_measurements_c.push_back(tracker_measurement_c);
        break;      
      }
    }
  }

  // Hungarian algorithm
  std::vector<std::vector<Scalar>> target_cost_matrix;
  std::vector<std::vector<Scalar>> tracker_cost_matrix;

  for (int i = 0; i < num_targets_; ++i) {
    std::vector<Scalar> place_holder_1(num_targets_, 1000.0);
    target_cost_matrix.push_back(place_holder_1);
  }

  for (int i = 0; i < num_trackers_; ++i) {
    std::vector<Scalar> place_holder_2(num_trackers_, 1000.0);
    tracker_cost_matrix.push_back(place_holder_2);
  }

  for (int i = 0; i < num_targets_; ++i) {
    for (int j = 0; j < target_measurements_w.size(); ++j) {
      Scalar target_cost = (target_kalman_filters_[i]->getEstimatedPosition() - target_measurements_w[j]).norm();
      target_cost_matrix[i][j] = target_cost;
    }
  }

  for (int i = 0; i < num_trackers_; ++i) {
    for (int j = 0; j < tracker_measurements_w.size(); ++j) {
      Scalar tracker_cost = (tracker_kalman_filters_[i]->getEstimatedPosition() - tracker_measurements_w[j]).norm();
      tracker_cost_matrix[i][j] = tracker_cost;
    }
  }

  // Hungarian matching algorithm for kalman filter update
	HungarianAlgorithm target_hungarian, tracker_hungarian;
	std::vector<int> target_assignment, tracker_assignment;
  
  Matrix<3, 3> R_C_W = rotateFromCameraToWorld();
  
  if (num_targets_ != 0) {
  	target_hungarian.Solve(target_cost_matrix, target_assignment);

    // 

    for (int i = 0; i < target_cost_matrix.size(); ++i) {
      if (target_cost_matrix[i][target_assignment[i]] > 900.0) {
        target_kalman_filters_[i]->predict();
      }
      else {
        target_kalman_filters_[i]->predict();
        target_kalman_filters_[i]->update(target_measurements_w[target_assignment[i]],
                                          target_measurements_c[target_assignment[i]],
                                          R_C_W);
      }
    }
  }
  
  if (num_trackers_ != 0) {
  	tracker_hungarian.Solve(tracker_cost_matrix, tracker_assignment);

    for (int i = 0; i < tracker_cost_matrix.size(); ++i) {
      if (tracker_cost_matrix[i][tracker_assignment[i]] > 900.0) {
        tracker_kalman_filters_[i]->predict();
      }
      else {
        tracker_kalman_filters_[i]->predict();
        tracker_kalman_filters_[i]->update(tracker_measurements_w[tracker_assignment[i]],
                                           tracker_measurements_c[tracker_assignment[i]],
                                           R_C_W);
      }
    }
  }



  //************************************************************************
  //*************************** Data Recoder *******************************
  //************************************************************************
  
  // // Record tracking data
  // // When saving tracking data, target and tracker position must be based on world frame
  // std::vector<Vector<3>> target_estim_pos, tracker_estim_pos;
  // std::vector<Matrix<3, 3>> target_cov, tracker_cov;

  // for (int i = 0; i < num_targets_; i++) {
  //   target_estim_pos.push_back(target_kalman_filters_[i]->getEstimatedPosition());
  //   target_cov.push_back(target_kalman_filters_[i]->getPositionErrorCovariance());
  // }
  // for (int i = 0; i < num_trackers_; i++) {
  //   tracker_estim_pos.push_back(tracker_kalman_filters_[i]->getEstimatedPosition());
  //   tracker_cov.push_back(tracker_kalman_filters_[i]->getPositionErrorCovariance());;
  // }

  // if (!tracking_save_->isFull()) {
  //   Vector<4> quadternion(quad_state_.x(QS::ATTW), quad_state_.x(QS::ATTX), quad_state_.x(QS::ATTY), quad_state_.x(QS::ATTZ));
  //   tracking_save_->store(quad_state_.p, quadternion, gt_target_positions_, target_estim_pos, target_cov, gt_tracker_positions_, tracker_estim_pos, tracker_cov, sim_dt_);
  // }
  // else if (tracking_flag_ && tracking_save_->isFull()) {
  //   tracking_save_->save();
  //   tracking_flag_ = false;
  //   std::cout << ">>> Tracker " << agent_id_ << "'s tracking output save is done" << std::endl;
  // }

  // // Record sensor data, just for first target case
  // for (int i = 0; i < target_positions.size(); i++) {
  //   detected = false;
  //   Vector<3> target_position = target_positions[i];
  //   for (int j = 0; j < num_cameras_; j++) {
  //     detected = multi_stereo_[j]->computePixelPoint(target_position, T_W_B);
  //     if (detected) {
  //       measured_position_ = multi_stereo_[j]->getSensorMeasurement();
  //       gt_pixels_ = multi_stereo_[j]->getGtPixels();
  //       pixels_ = multi_stereo_[j]->getPixels();
  //       break;      
  //     }
  //   }
  // }

  // if (!sensor_save_->isFull()) {
  //   Vector<3> gt_target_position_in_camera = fromWorldToCamera(gt_target_positions_[0]);
  //   sensor_save_->store(gt_pixels_, pixels_, gt_target_position_in_camera, measured_position_, sim_dt_);
  // }
  // if (sensor_flag_ && sensor_save_->isFull()) {
  //   sensor_save_->save();
  //   sensor_flag_ = false;
  //   std::cout << ">>> Sensor output save is done" << std::endl;
  // }

  //************************************************************************
  //*************************** Data Recoder *******************************
  //************************************************************************



  // Simulate quadrotor (apply rungekutta4th 8 times during 0.02s)
  tracker_ptr_->run(cmd_, sim_dt_);

  // Update observations
  getObs(obs);

  // Reward function of tracker
  Scalar reward = rewardFunction();

  return reward;
}

bool TrackerQuadrotorEnv::getObs(Ref<Vector<>> obs)
{
  tracker_ptr_->getState(&quad_state_);

  //
  for (int i = 0; i < num_targets_; ++i) {
    estimated_target_positions_.push_back(target_kalman_filters_[i]->getEstimatedPosition());
    estimated_target_velocities_.push_back(target_kalman_filters_[i]->getEstimatedVelocity());
    estimated_target_ranges_.push_back(computeEuclideanDistance(quad_state_.p, estimated_target_positions_[i]));
  }
  // for (int i = 0; i < num_trackers_; ++i) {
  //   estimated_tracker_positions_.push_back(tracker_kalman_filters_[i]->getEstimatedPosition());
  //   estimated_tracker_velocities_.push_back(tracker_kalman_filters_[i]->getEstimatedVelocity());
  //   estimated_tracker_ranges_.push_back(computeEuclideanDistance(quad_state_.p, estimated_tracker_positions_[i]));
  // }

  std::vector<Scalar> gt_tracker_ranges;
  for (int i = 0; i < num_trackers_; ++i) {
    gt_tracker_ranges.push_back(computeEuclideanDistance(quad_state_.p, gt_tracker_positions_[i]));
  }

  Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());
  
  // Ego oservation dim: 3 + 3 + 9 + 3 + 1 = 19
  // Target observation dim: 3 + 3 + 1 + 1 + 1 = 9 (+1 id in multi)
  // // Other tracker observations 3 + 1 + 1 + 1 = 6 (+1 id in multi)

  // //************************************************************************
  // // Control policy test
  // quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w, radius_,
  //              estimated_target_positions_[0], estimated_target_velocities_[0], radius_, estimated_target_ranges_[0], radius_ * 2;



  // //************************************************************************
  // // Single target tracking (2 targets)
  // quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w, radius_,
  //              estimated_target_positions_[0], estimated_target_velocities_[0], radius_, estimated_target_ranges_[0], radius_ * 2, 
  //              estimated_target_positions_[1], estimated_target_velocities_[1], radius_, estimated_target_ranges_[1], radius_ * 2;



  // //************************************************************************
  // // Single target tracking (4 targets)
  // quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w, radius_,
  //              estimated_target_positions_[0], estimated_target_velocities_[0], radius_, estimated_target_ranges_[0], radius_ * 2, 
  //              estimated_target_positions_[1], estimated_target_velocities_[1], radius_, estimated_target_ranges_[1], radius_ * 2,
  //              estimated_target_positions_[2], estimated_target_velocities_[2], radius_, estimated_target_ranges_[2], radius_ * 2,
  //              estimated_target_positions_[3], estimated_target_velocities_[3], radius_, estimated_target_ranges_[3], radius_ * 2;



  //************************************************************************
  // Multi target tracking (2 agents & 3 targets)
  quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w, radius_,
               estimated_target_positions_[0], estimated_target_velocities_[0], radius_, estimated_target_ranges_[0], radius_ * 2, 0,
               estimated_target_positions_[1], estimated_target_velocities_[1], radius_, estimated_target_ranges_[1], radius_ * 2, 0,
               estimated_target_positions_[2], estimated_target_velocities_[2], radius_, estimated_target_ranges_[2], radius_ * 2, 0,
               
               gt_tracker_positions_[0], radius_, gt_tracker_ranges[0], radius_ * 2, 1;



  // //************************************************************************
  // // Multi target tracking (3 agents & 4 targets)
  // quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w, radius_,
  //              estimated_target_positions_[0], estimated_target_velocities_[0], radius_, estimated_target_ranges_[0], radius_ * 2, 0,
  //              estimated_target_positions_[1], estimated_target_velocities_[1], radius_, estimated_target_ranges_[1], radius_ * 2, 0,
  //              estimated_target_positions_[2], estimated_target_velocities_[2], radius_, estimated_target_ranges_[2], radius_ * 2, 0,
  //              estimated_target_positions_[3], estimated_target_velocities_[3], radius_, estimated_target_ranges_[3], radius_ * 2, 0,
               
  //              gt_tracker_positions_[0], radius_, gt_tracker_ranges[0], radius_ * 2, 1,
  //              gt_tracker_positions_[1], radius_, gt_tracker_ranges[1], radius_ * 2, 1;


  //************************************************************************
  // obs.segment<28>(0) = quad_obs_; // control policy test

  // obs.segment<37>(0) = quad_obs_; // single (2 targets)
  // obs.segment<55>(0) = quad_obs_; // single (4 targets)
  
  obs.segment<56>(0) = quad_obs_; // multi (2 agents & 3 targets)
  // obs.segment<73>(0) = quad_obs_; // multi (3 agents & 4 targets)

  return true;
}

// Scalar TrackerQuadrotorEnv::rewardFunction()
// {
//   // Outter coefficient
//   Scalar range = computeEuclideanDistance(quad_state_.p, gt_target_positions_[0]);

//   //
//   Scalar c1 = 2.0;
//   Scalar c2 = 0.1;

//   // Progress reward
//   Scalar progress_reward = 0.0;
//   if (first_) {
//     first_ = false;
//   }
//   else {
//     progress_reward = prev_range_ - range;
//   }

//   // Heading reward
//   Vector<3> h = quad_state_.q().toRotationMatrix() * Vector<3>(1, 0, 0); // Ego tracker heading vector
//   h = h / (h.norm() + 1e-8);
//   Vector<3> d = gt_target_positions_[0] - quad_state_.p; // Relative distance to target
//   d = d / (d.norm() + 1e-8);

//   Scalar dot_value = h.dot(d);
//   dot_value = std::max(static_cast<Scalar>(-0.999), std::min(static_cast<Scalar>(0.999), dot_value));
//   Scalar theta = acos(dot_value);
//   Scalar heading_reward = exp(-10.0 * pow(theta, 3));

//   prev_range_ = range;

//   Scalar total_reward = c1 * progress_reward + c2 * heading_reward;

//   // std::cout << "-------------------------------------" << std::endl;
//   // std::cout << "progress reward : " << progress_reward << std::endl;
//   // std::cout << "heading reward  : " << heading_reward << std::endl;
//   // std::cout << "total reward    : " << total_reward << std::endl;

//   return total_reward;
// }

Scalar TrackerQuadrotorEnv::rewardFunction()
{
  // Outter coefficient
  Scalar c1 = 1.0;
  Scalar c2 = 0.3;
  Scalar c3 = -1e-4;

  // Covariance reward
  Scalar cov_reward = 0.0;
  Scalar avg_cov_det = 0.0;
  std::vector<Scalar> cov_list;
  for (int i = 0; i < num_targets_; ++i) {
    Matrix<3, 3> cov = target_kalman_filters_[i]->getPositionErrorCovariance();
    Scalar cov_det = cov.determinant();
    Scalar scaled_cov_det = cov_det * 9; // 9 is scale factor
    cov_list.push_back(scaled_cov_det);
    avg_cov_det += exp(-0.1 * pow(scaled_cov_det, 2));
  }
  cov_reward = avg_cov_det / num_targets_;

  // Heading reward
  Scalar heading_reward = 0.0;
  Scalar avg_heading = 0.0;
  Vector<3> h = quad_state_.q().toRotationMatrix() * Vector<3>(1, 0, 0); // Ego tracker heading vector
  h = h / (h.norm() + 1e-8);
  for (int i = 0; i < num_targets_; ++i) {
    Vector<3> target_position = target_kalman_filters_[i]->getEstimatedPosition();
    Vector<3> d = target_position - quad_state_.p; // Relative distance to target
    d = d / (d.norm() + 1e-8);

    Scalar dot_value = h.dot(d);
    dot_value = std::max(static_cast<Scalar>(-0.999), std::min(static_cast<Scalar>(0.999), dot_value));
    Scalar theta = acos(dot_value);

    if (std::isnan(theta)) {
      std::cout << "nan occurs from individual theta" << std::endl;
      std::cout << "theta : " << theta << std::endl;
      std::cout << "dot_value : " << dot_value << std::endl;
      std::cout << "h : " << h << std::endl;
      std::cout << "d : " << d << std::endl;
      exit(0);
    }

    avg_heading += exp(-10.0 * pow(theta, 3));
  }
  heading_reward = avg_heading / num_targets_;


  // Smooth action reward (penalty)
  Scalar cmd_reward = pow((quad_act_ - prev_act_).norm(), 2);
  prev_act_ = quad_act_;

  Scalar total_reward = c1 * cov_reward + c2 * heading_reward + c3 * cmd_reward;

  // std::cout << "-------------------------------------" << std::endl;
  // std::cout << "cov det        : " << cov_list[0] << ", " << cov_list[1] << ", " << cov_list[2] << ", " << cov_list[3] << std::endl;
  // std::cout << "avg cov det    : " << avg_cov_det << std::endl;
  // std::cout << "cov reward     : " << c1 * cov_reward << std::endl;
  // std::cout << "heading reward : " << c2 * heading_reward << std::endl;
  // std::cout << "cmd reward     : " << c3 * cmd_reward << std::endl;
  // std::cout << "total reward   : " << total_reward << std::endl;

  return total_reward;
}

Scalar TrackerQuadrotorEnv::computeEuclideanDistance(Ref<Vector<3>> p1, Ref<Vector<3>> p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

bool TrackerQuadrotorEnv::isTerminalState(Scalar &reward) {
  // Out of the world
  if (quad_state_.x(QS::POSZ) <= 0.02  || quad_state_.x(QS::POSZ) >= 49.0 ||
      quad_state_.x(QS::POSX) <= -49.0 || quad_state_.x(QS::POSX) >= 49.0 ||
      quad_state_.x(QS::POSY) <= -49.0 || quad_state_.x(QS::POSY) >= 49.0) {
    reward = -5.0;
    return true;
  }



  // //************************************************************************
  // //************************** Control Policies ****************************
  // //************************************************************************

  // for (int i = 0; i < num_targets_; ++i) {
  //   Scalar distance = computeEuclideanDistance(quad_state_.p, gt_target_positions_[i]);
  //   if (distance <= 1.5) {
  //     reward = 10.0;
  //     return true;
  //   }
  // }

  // //************************************************************************
  // //************************** Control Policies ****************************
  // //************************************************************************



  // Collision to target or tracker
  for (int i = 0; i < num_targets_; ++i) {
    Scalar distance = computeEuclideanDistance(quad_state_.p, gt_target_positions_[i]);
    if (distance <= 0.7) {
      reward = -5.0;
      return true;
    }
  }
  
  for (int i = 0; i < num_trackers_; ++ i) {
    Scalar distance = computeEuclideanDistance(quad_state_.p, gt_tracker_positions_[i]);
    if (distance <= 0.7) {
      reward = -5.0;
      return true;
    }
  }

  reward = 0.0;
  return false;
}

bool TrackerQuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    kp_vxy_ = cfg["quadrotor_pid_controller_gain"]["kp_vxy"].as<Scalar>();
    ki_vxy_ = cfg["quadrotor_pid_controller_gain"]["ki_vxy"].as<Scalar>();
    kd_vxy_ = cfg["quadrotor_pid_controller_gain"]["kd_vxy"].as<Scalar>();
    kp_vz_ = cfg["quadrotor_pid_controller_gain"]["kp_vz"].as<Scalar>();
    ki_vz_ = cfg["quadrotor_pid_controller_gain"]["ki_vz"].as<Scalar>();
    kd_vz_ = cfg["quadrotor_pid_controller_gain"]["kd_vz"].as<Scalar>();
    kp_angle_ = cfg["quadrotor_pid_controller_gain"]["kp_angle"].as<Scalar>();
    ki_angle_ = cfg["quadrotor_pid_controller_gain"]["ki_angle"].as<Scalar>();
    kd_angle_ = cfg["quadrotor_pid_controller_gain"]["kd_angle"].as<Scalar>();
    kp_wz_ = cfg["quadrotor_pid_controller_gain"]["kp_wz"].as<Scalar>();
    ki_wz_ = cfg["quadrotor_pid_controller_gain"]["ki_wz"].as<Scalar>();
    kd_wz_ = cfg["quadrotor_pid_controller_gain"]["kd_wz"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

bool TrackerQuadrotorEnv::getAct(Ref<Vector<>> act) const
{
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool TrackerQuadrotorEnv::getAct(Command *const cmd) const
{
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void TrackerQuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge)
{
  bridge->addTracker(tracker_ptr_);
}

}  // namespace flightlib
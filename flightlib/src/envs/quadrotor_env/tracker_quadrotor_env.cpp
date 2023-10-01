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

  // Mount front, left, right stereo cameras
  for (int i = 0; i < num_cameras_; i++) {
    multi_stereo_.push_back(std::make_shared<StereoCamera>());
  }

  Vector<3> d_l = Vector<3>(0.06, -0.015, -0.1); // from left camera frame to body
  Vector<3> d_r = Vector<3>(-0.06, -0.015, -0.1); // from right camera frame to body
  Matrix<3, 3> R_front = Rot_x(-M_PI_2) * Rot_z(-M_PI_2);
  Matrix<3, 3> R_left  = Rot_y(-2.0/3.0 * M_PI) * Rot_x(-M_PI_2) * Rot_z(-M_PI_2);
  Matrix<3, 3> R_right = Rot_y(2.0/3.0 * M_PI) * Rot_x(-M_PI_2) * Rot_z(-M_PI_2);
  multi_stereo_[0]->init(d_l, d_r, R_front); // front camera
  multi_stereo_[1]->init(d_l, d_r, R_left); // left back camera
  multi_stereo_[2]->init(d_l, d_r, R_right); // right back camera

  // Data recoder
  sensor_save_ = SensorSave();
  tracking_save_ = TrackingSave();

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  tracker_ptr_->updateDynamics(dynamics);
  tracker_ptr_->setVelocityPIDGain(kp_vxy_, ki_vxy_, kd_vxy_, kp_vz_, ki_vz_, kd_vz_, kp_angle_, ki_angle_, kd_angle_, kp_wz_, ki_wz_, kd_wz_);

  // define a bounding box
  world_box_ << -100, 100, -100, 100, 0, 100;
  if (!tracker_ptr_->setWorldBox(world_box_))
  {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  // obs_dim_ = trackerquadenv::kNObs;
  obs_dim_ = 22;
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
  // obtain observations
  getObs(obs);
  return true;
}

bool TrackerQuadrotorEnv::reset(Ref<Vector<>> obs, Ref<Vector<>> position,
                                const std::vector<Vector<3>>& target_positions, const std::vector<Vector<3>>& tracker_positions) {
  quad_state_.setZero();
  quad_act_.setZero();

  quad_state_.x(QS::POSX) = position[0];
  quad_state_.x(QS::POSY) = position[1];
  quad_state_.x(QS::POSZ) = position[2];

  // In order of yaw, pitch, roll
  // Scalar yaw = uniform_dist_(random_gen_) * M_PI;
  Scalar yaw = M_PI_2;
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

  // Initialize multi kalman filter for target
  for (int i = 0; i < num_targets_; ++i){
    std::shared_ptr<KalmanFilter> target_kf = std::make_shared<KalmanFilter>();
    Vector<6> x0 = (Vector<6>() << target_positions[i][0], 0, target_positions[i][1], 0, target_positions[i][2], 0).finished();
    target_kf->init(sim_dt_, x0);
    target_kalman_filters_.push_back(target_kf);
  }

  // Initialize multi kalman filter for tracker
  for (int i = 0; i < num_trackers_; ++i){
    std::shared_ptr<KalmanFilter> tracker_kf = std::make_shared<KalmanFilter>();
    Vector<6> x0 = (Vector<6>() << tracker_positions[i][0], 0, tracker_positions[i][1], 0, tracker_positions[i][2], 0).finished();
    tracker_kf->init(sim_dt_, x0);
    tracker_kalman_filters_.push_back(tracker_kf);
  }

  // Initialize tracking recoder
  tracking_save_.init(num_targets_, num_trackers_);

  // Reset velocity control command
  cmd_.t = 0.0;
  cmd_.velocity.setZero();

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
  cmd_.velocity = quad_act_;

  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_W_B = T_B_W.inverse(); // World to body

  // Store ground truth position
  gt_target_positions_ = target_positions;
  gt_tracker_positions_ = tracker_positions;

  // Try to detect the number of "n" targets
  bool detected = false;
  std::vector<Vector<3>> target_measurements;

  for (int i = 0; i < target_positions.size(); i++) {
    detected = false;
    Vector<3> tracker_position = target_positions[i];
    for (int j = 0; j < num_cameras_; j++) {
      detected = multi_stereo_[j]->computePixelPoint(tracker_position, T_W_B);
      if (detected) {
        Vector<3> target_measurement = multi_stereo_[j]->getObjectPosition();
        target_measurements.push_back(target_measurement);
        break;      
      }
    }
  }
  
  // Try to detect the number of "m-1" trackers
  std::vector<Vector<3>> tracker_measurements;

  for (int i = 0; i < tracker_positions.size(); i++) {
    detected = false;
    Vector<3> tracker_position = tracker_positions[i];
    for (int j = 0; j < num_cameras_; j++) {
      detected = multi_stereo_[j]->computePixelPoint(tracker_position, T_W_B);
      if (detected) {
        Vector<3> tracker_measurement = multi_stereo_[j]->getObjectPosition();
        tracker_measurements.push_back(tracker_measurement);
        break;      
      }
    }
  }

  std::cout << "-------------------------------------" << std::endl;
  std::cout << "Tracker has " << target_measurements.size() << " target measurements" << std::endl;
  for (int i = 0; i < target_measurements.size(); i++) {
    std::cout << target_measurements[i][0] << ", " << target_measurements[i][1] << ", " << target_measurements[i][2] << std::endl;
  }
  std::cout << "Tracker has " << tracker_measurements.size() << " tracker measurements" << std::endl;
  for (int i = 0; i < tracker_measurements.size(); i++) {
    std::cout << tracker_measurements[i][0] << ", " << tracker_measurements[i][1] << ", " << tracker_measurements[i][2] << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;
  
  // Hungarian algorithm
  // Matrix<> target_cost_matrix = Matrix<>::Ones(num_targets_, num_targets_) * 1000.0;
  // Matrix<> tracker_cost_matrix = Matrix<>::Ones(num_trackers_, num_trackers_) * 1000.0;

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
    for (int j = 0; j < target_measurements.size(); ++j) {
      Scalar target_cost = (target_kalman_filters_[i]->getEstimatedPosition() - target_measurements[j]).norm();
      target_cost_matrix[i][j] = target_cost;
    }
  }

  for (int i = 0; i < num_trackers_; ++i) {
    for (int j = 0; j < tracker_measurements.size(); ++j) {
      Scalar tracker_cost = (tracker_kalman_filters_[i]->getEstimatedPosition() - tracker_measurements[j]).norm();
      tracker_cost_matrix[i][j] = tracker_cost;
    }
  }

  // Hungarian matching algorithm for kalman filter update
	HungarianAlgorithm target_hungarian, tracker_hungarian;
	std::vector<int> target_assignment, tracker_assignment;
  
  
  if (num_targets_ != 0) {
  	target_hungarian.Solve(target_cost_matrix, target_assignment);

    for (int i = 0; i < target_cost_matrix.size(); ++i) {
      if (target_cost_matrix[i][target_assignment[i]] > 900.0) {
        std::cout << i << " kalman filter starts prediction" << std::endl;
        target_kalman_filters_[i]->predict();
      }
      else {
        std::cout << i << " kalman filter starts update" << std::endl;
        Vector<3> temp = target_measurements[target_assignment[i]];
        std::cout << "measurement " << temp[0] << ", " << temp[1] << ", " << temp[2] << " is equal to ";
        Vector<3> temp2 = target_kalman_filters_[i]->getEstimatedPosition();
        std::cout << temp2[0] << ", " << temp2[1] << ", " << temp2[2];
        std::cout << "\tcost:" << (temp - temp2).norm() << std::endl;
        target_kalman_filters_[i]->predict();
        target_kalman_filters_[i]->update(target_measurements[target_assignment[i]]);
      }
    }
  }
  
  if (num_trackers_ != 0) {
  	tracker_hungarian.Solve(tracker_cost_matrix, tracker_assignment);

    for (int i = 0; i < tracker_cost_matrix.size(); ++i) {
      if (tracker_cost_matrix[i][tracker_assignment[i]] > 900.0) {
        std::cout << i << " kalman filter starts prediction" << std::endl;
        tracker_kalman_filters_[i]->predict();
      }
      else {
        std::cout << i << " kalman filter starts update" << std::endl;
        tracker_kalman_filters_[i]->predict();
        tracker_kalman_filters_[i]->update(tracker_measurements[tracker_assignment[i]]);
      }
    }
  }

  // Record tracking data
  std::vector<Vector<3>> target_estim_pos, tracker_estim_pos;
  std::vector<Matrix<6, 6>> target_cov, tracker_cov;

  for (int i = 0; i < num_targets_; i++) {
    target_estim_pos.push_back(target_kalman_filters_[i]->getEstimatedPosition());
    target_cov.push_back(target_kalman_filters_[i]->getErrorCovariance());
  }
  for (int i = 0; i < num_trackers_; i++) {
    tracker_estim_pos.push_back(tracker_kalman_filters_[i]->getEstimatedPosition());
    tracker_cov.push_back(tracker_kalman_filters_[i]->getErrorCovariance());;
  }

  if (!tracking_save_.isFull())
    tracking_save_.store(quad_state_.p, gt_target_positions_, target_estim_pos, target_cov, gt_tracker_positions_, tracker_estim_pos, tracker_cov, sim_dt_);
  else if (tracking_flag_ && tracking_save_.isFull()) {
    tracking_save_.save();
    tracking_flag_ = false;
    std::cout << ">>> Tracking output save is done" << std::endl;
  }

  // Vector<3> estimated_position = kf_->computeEstimatedPositionWrtWorld(T_LC_W);
  // Scalar estimated_range = kf_->computeRangeWrtBody(quad_state_.p, T_LC_B);
  // Matrix<6, 6> covariance = kf_->getErrorCovariance();

  // if (sensor_flag_)
  //   sensor_save_.store(gt_pixels_, pixels_, gt_target_position_, estimated_position_, sim_dt_);
  // if (sensor_flag_ && sensor_save_.isFull()) {
  //   sensor_save_.save();
  //   sensor_flag_ = false;
  //   std::cout << ">>> Sensor output save is done" << std::endl;
  // }

  // Simulate quadrotor (apply rungekutta4th 8 times during 0.02s)
  tracker_ptr_->run(cmd_, sim_dt_);

  // Update observations
  getObs(obs);

  // Reward function of tracker
  // Scalar reward = rewardFunction(target_position);
  Scalar reward = 0.0;

  return reward;
}

bool TrackerQuadrotorEnv::getObs(Ref<Vector<>> obs)
{
  tracker_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);

  // Matrix<4, 4> T_B_W = getBodyToWorld();
  // Matrix<4, 4> T_LC_B = stereo_camera_->getFromLeftCameraToBody();
  // Matrix<4, 4> T_LC_W = T_B_W * T_LC_B;
  // Vector<3> target_p = kf_->computeEstimatedPositionWrtWorld(T_LC_W);
  // Scalar target_r = kf_->computeRangeWrtBody(quad_state_.p, T_LC_B);


  // Vector<3> gt_relative_position(gt_target_position_[0] - quad_state_.x(QS::POSX),
  //                                gt_target_position_[1] - quad_state_.x(QS::POSY),
  //                                gt_target_position_[2] - quad_state_.x(QS::POSZ));

  Vector<3> gt_relative_position(1, 1, 1);
  Scalar gt_range = sqrt(pow(gt_relative_position[0], 2) + pow(gt_relative_position[1], 2) + pow(gt_relative_position[2], 2));

  // // observation dim : 3 + 9 + 3 = 15
  // obs.segment<quadenv::kNObs>(quadenv::kObs) << quad_state_.p, ori, quad_state_.v;
  Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());

  // Observation dim: 3 + 3 + 9 + 3 = 18
  quad_obs_ << quad_state_.p, quad_state_.v, ori, quad_state_.w,
               gt_relative_position, gt_range; // target information

  obs.segment<22>(0) = quad_obs_;

  return true;
}

Scalar TrackerQuadrotorEnv::rewardFunction(Vector<3> target_position)
{
  // Outter coefficient
  Scalar c1 = 10.0;
  Scalar c2 = 1.0;
  Scalar c3 = 0.5;
  Scalar c4 = -1e-3;

  // Inner coefficient
  Scalar i1 = -0.1;
  Scalar i2 = -0.01;
  Scalar i3 = -10.0;

  // Scalar range_xy = hypot(gt_target_position_[0] - quad_state_.x(QS::POSX), gt_target_position_[1] - quad_state_.x(QS::POSY));
  // Scalar range_z = abs(gt_target_position_[2] - quad_state_.x(QS::POSZ));

  Scalar range_xy = 0.0;
  Scalar range_z = 0.0;

  // // xy range reward
  // Scalar range_xy_reward = exp(i1 * pow(range_xy, 3));

  // Progress reward
  Scalar progress_reward = prev_range_ - range_xy;

  if (first_) {
    first_ = false;
    progress_reward = 0.0;
  }
  prev_range_ = range_xy;

  // z range reward
  Scalar range_z_reward = exp(i2 * pow(range_z, 3));

  // Perception reward
  Vector<3> h = quad_state_.q().toRotationMatrix() * Vector<3>(1, 0, 0);
  // Vector<3> d(gt_target_position_[0] - quad_state_.x(QS::POSX), gt_target_position_[1] - quad_state_.x(QS::POSY), gt_target_position_[2] - quad_state_.x(QS::POSZ));
  Vector<3> d(1, 1, 1);
  h = h / h.norm();
  d = d / d.norm();

  Scalar theta = acos(h.dot(d));
  Scalar perception_reward = exp(i3 * pow(theta, 3));
 
  // command reward
  Scalar command_reward = pow((quad_act_ - prev_act_).norm(), 2);


  // prev_act_ = quad_act_;
  // prev_range_ = range;

  // std::cout << "progress   : " << progress_reward << std::endl;
  // std::cout << "range z    : " << range_z_reward << std::endl;
  // std::cout << "perception : " << perception_reward << std::endl;
  // std::cout << "command        : " << command_reward << std::endl;
  // std::cout << "scaled command : " << c4 * command_reward << std::endl;


  // Scalar total_reward = c1 * range_xy_reward + c2 * range_z_reward + c3 * perception_reward + c4 * command_reward;
  Scalar total_reward = c1 * progress_reward + c2 * range_z_reward + c3 * perception_reward;

  return total_reward;
}

bool TrackerQuadrotorEnv::isTerminalState(Scalar &reward) {
  // Out of the world
  if (quad_state_.x(QS::POSZ) <= 0.02  || quad_state_.x(QS::POSZ) >= 30.0 ||
      quad_state_.x(QS::POSX) <= -30.0 || quad_state_.x(QS::POSX) >= 30.0 ||
      quad_state_.x(QS::POSY) <= -30.0 || quad_state_.x(QS::POSY) >= 30.0) {
    reward = -5.0;
    return true;
  }

  // // Clashing target
  // Scalar gt_range =  sqrt(pow(quad_state_.x(QS::POSX) - gt_target_position_[0], 2)
  //                       + pow(quad_state_.x(QS::POSY) - gt_target_position_[1], 2)
  //                       + pow(quad_state_.x(QS::POSZ) - gt_target_position_[2], 2));
  // if (gt_range <= 0.5) {
  //   reward = 2.0;
  //   return true;
  // }

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
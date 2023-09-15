#include "flightlib/envs/quadrotor_env/tracker_quadrotor_env.hpp"

namespace flightlib {

TrackerQuadrotorEnv::TrackerQuadrotorEnv() : TrackerQuadrotorEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/tracker_quadrotor_env.yaml")) {}
TrackerQuadrotorEnv::TrackerQuadrotorEnv(const std::string &cfg_path) : EnvBase()
{
  // load configuration file
  // YAML::Node cfg_ = YAML::LoadFile(cfg_path);
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  // load parameters
  loadParam(cfg_);

  tracker_ptr_ = std::make_shared<TrackerQuadrotor>();
  stereo_camera_ = std::make_shared<StereoCamera>();
  sensor_save_ = SensorSaveV2();
  kf_ = std::make_shared<KalmanFilter>();
  tracking_save_ = TrackingSaveV2();

  maf_ = MovingAverageFilter(20);

  // Initialize kalman filter
  Vector<9> x0 = (Vector<9>() << 5, 0, 0, 0, 0, 0, 0, 0, 0).finished(); // w.r.t. camera frame
  Matrix<9, 9> P0 = 1e2 * Matrix<9, 9>::Identity();
  Scalar f = stereo_camera_->getFocalLength();
  Scalar c = stereo_camera_->getPrincipalPoint();
  Scalar b = stereo_camera_->getBaseline();
  kf_->init(sim_dt_, x0, P0, 1.0, 20);

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

bool TrackerQuadrotorEnv::reset(Ref<Vector<>> obs, const bool random)
{
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
    // quad_state_.x(QS::POSX) = uniform_dist_(random_gen_) * 8.0;
    // quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * 8.0;
    // quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) * 2.0 + 5.0;

    // if (quad_state_.x(QS::POSX) < -8.0)
    //   quad_state_.x(QS::POSX) = -8.0;
    // else if (quad_state_.x(QS::POSX) > 8.0)
    //   quad_state_.x(QS::POSX) = 8.0;

    // if (quad_state_.x(QS::POSY) < -8.0)
    //   quad_state_.x(QS::POSY) = -8.0;
    // else if ((quad_state_.x(QS::POSY) > 8.0))
    //   quad_state_.x(QS::POSY) = 8.0;

    // if (quad_state_.x(QS::POSZ) < 1.0)
    //   quad_state_.x(QS::POSZ) = 1.0;
    // else if (quad_state_.x(QS::POSZ) > 10.0)
    //   quad_state_.x(QS::POSZ) = 10.0;
    
    quad_state_.x(QS::POSX) = 0.0;
    quad_state_.x(QS::POSY) = -3.0;
    quad_state_.x(QS::POSZ) = 5.0;


    // In order of yaw, pitch, roll
    Vector<3> euler(M_PI_2, 0, 0);
    Vector<4> quaternion = eulerToQuaternion(euler);

    quad_state_.x(QS::ATTW) = quaternion[0];
    quad_state_.x(QS::ATTX) = quaternion[1];
    quad_state_.x(QS::ATTY) = quaternion[2];
    quad_state_.x(QS::ATTZ) = quaternion[3];
  }
  // Reset quadrotor with random states
  tracker_ptr_->reset(quad_state_);

  // Reset tracking algorithm
  kf_->reset();

  // Reset velocity control command
  cmd_.t = 0.0;
  cmd_.velocity.setZero();

  // // Reset mass-normalized collective thrust & body rates control command
  // cmd_.t = 0.0;
  // cmd_.collective_thrust = 0.0;
  // cmd_.omega.setZero();

  gt_target_point_ = Vector<3>(0, 0, 5);
  t_b_ =  Vector<3>(5, 0, 0);

  maf_.reset();


  // obtain observations
  getObs(obs);
  return true;
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

bool TrackerQuadrotorEnv::getObs(Ref<Vector<>> obs)
{
  tracker_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;


  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_LC_B = stereo_camera_->getFromLeftCameraToBody();
  Matrix<4, 4> T_LC_W = T_B_W * T_LC_B;
  Vector<3> target_p = kf_->computeEstimatedPositionWrtWorld(T_LC_W);
  Scalar target_r = kf_->computeRangeWrtBody(quad_state_.p, T_LC_B);

  Scalar gt_r =  sqrt(pow(quad_state_.x(QS::POSX) - gt_target_point_[0], 2)
                        + pow(quad_state_.x(QS::POSY) - gt_target_point_[1], 2)
                        + pow(quad_state_.x(QS::POSZ) - gt_target_point_[2], 2));

  Scalar test_r =  sqrt(pow(t_b_[0], 2) + pow(t_b_[1], 2) + pow(t_b_[2], 2));

  // std::cout << "Check: " << gt_r << ", " << test_r << std::endl;

  Matrix<3, 3> rotation = quad_state_.q().toRotationMatrix();

  quad_obs_ << quad_state_.p, // tracker state (15 dim)
               quad_state_.v,
               rotation(0,0), rotation(0,1), rotation(0,2),
               rotation(1,0), rotation(1,1), rotation(1,2),
               rotation(2,0), rotation(2,1), rotation(2,2),
               quad_state_.w,
               t_b_, // target information
               gt_r; 

  obs.segment<22>(0) = quad_obs_;

  return true;
}

Scalar TrackerQuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs)
{
  // Reward function of tracker quadrotor
  Scalar total_reward = 0.0;
  return total_reward;
}

Scalar TrackerQuadrotorEnv::rewardFunction(const Scalar range)
{
  // // Test reward
  // if (range < 1.0)
  //   return -1.0;
  // else
  //   return 10.0 * exp(-range + 1);

  // Coefficient
  Scalar c1 = 2.0;
  Scalar c2 = 0.5;
  Scalar c3 = -5.0;
  Scalar c4 = -1e-4;

  // Progress reward
  Scalar progress_reward;
  if (range < 1.0)
    progress_reward = -1.0;
  else
    progress_reward = exp(-range + 1);

  // Perception reward
  Vector<3> h = quad_state_.q().toRotationMatrix() * Vector<3>(1, 0, 0);
  Vector<3> d(gt_target_point_[0] - quad_state_.x(QS::POSX), gt_target_point_[1] - quad_state_.x(QS::POSY), gt_target_point_[2] - quad_state_.x(QS::POSZ));
  h = h / h.norm();
  d = d / d.norm();

  Scalar theta = acos(h.dot(d));
  // Scalar perception_reward = exp(c3 * theta);
  Scalar perception_reward = exp(c3 * pow(theta, 4));
 
  // command reward
  Scalar command_reward = pow((quad_act_ - prev_act_).norm(), 2);

  // std::cout << ">>> progress: " << progress_reward << std::endl;
  // std::cout << ">>> perception: " << perception_reward << std::endl;
  // std::cout << ">>> command: " << command_reward << std::endl;

  prev_act_ = quad_act_;

  Scalar total_reward = c1 * progress_reward + c2 * perception_reward + c4 * command_reward;
  // std::cout << ">>> Total reward: " << total_reward << std::endl;
  return total_reward;


  // Scalar moving_average = maf_.add(range);
  // std::cout << ">>> moving_average: " << moving_average << std::endl;

  // if (range <= 1.5)
  //   return 1.0;

  // if (range <= moving_average)
  //   return 1.0;
  // else
  //   return -1.0;
}

Scalar TrackerQuadrotorEnv::trackerStep(const Ref<Vector<>> act, Ref<Vector<>> obs, Vector<3> target_point)
{
  quad_act_ = act;
  cmd_.t += sim_dt_;
  
  cmd_.velocity = quad_act_;

  // cmd_.collective_thrust = act[0];
  // cmd_.omega = act.segment<3>(1);;


  // Simulate quadrotor (apply rungekutta4th 8 times during 0.02s)
  tracker_ptr_->run(cmd_, sim_dt_);

  // get target
  Matrix<4, 4> T_B_W = getBodyToWorld();
  Matrix<4, 4> T_W_B = T_B_W.inverse(); // World to body




  gt_target_point_ = target_point;
  // Transform target position from world to body
  Vector<4> T_W(target_point[0], target_point[1], target_point[2], 1);
  Vector<3> t_b = T_W_B.topRows<3>() * T_W;
  t_b_ = t_b;




  kf_->predict();
  if (stereo_camera_->processImagePoint(target_point, T_W_B))
  {
    gt_pixels_ = stereo_camera_->getGroundTruthPixels();
    gt_target_position_ = stereo_camera_->getGroundTruthPosition();
    pixels_ = stereo_camera_->getPixels();
    target_position_ = stereo_camera_->getTargetPosition();

    kf_->update(target_position_);
  }
  // else
  // {
  //   std::cout << ">>> Impossible to detect target" << std::endl;
  // }

  Matrix<4, 4> T_LC_B = stereo_camera_->getFromLeftCameraToBody();
  Matrix<4, 4> T_LC_W = T_B_W * T_LC_B;
  Vector<3> estimated_position = kf_->computeEstimatedPositionWrtWorld(T_LC_W);
  Scalar estimated_range = kf_->computeRangeWrtBody(quad_state_.p, T_LC_B);
  Matrix<9, 9> covariance = kf_->getErrorCovariance();

  Scalar gt_range =  sqrt(pow(quad_state_.x(QS::POSX) - target_point[0], 2)
                        + pow(quad_state_.x(QS::POSY) - target_point[1], 2)
                        + pow(quad_state_.x(QS::POSZ) - target_point[2], 2));

  if (sensor_flag_)
    sensor_save_.store(gt_pixels_, pixels_, gt_target_position_, target_position_, sim_dt_);
  if (sensor_flag_ && sensor_save_.isFull()) {
    sensor_save_.save();
    sensor_flag_ = false;
    std::cout << ">>> Sensor output save is done" << std::endl;
  }

  // Kalman filter output
  if (tracking_flag_)
    tracking_save_.store(target_point, estimated_position, covariance, sim_dt_);
  if (tracking_flag_ && tracking_save_.isFull()) {
    tracking_save_.save();
    tracking_flag_ = false;
    std::cout << ">>> Tracking output save is done" << std::endl;
  }

  // Update observations
  getObs(obs);

  // Reward function of tracker quadrotor
  // Scalar reward = rewardFunction(estimated_range);
  Scalar reward = rewardFunction(gt_range);

  // std::cout << ">>> reward: " << reward << std::endl;

  return reward;
}

bool TrackerQuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02  || quad_state_.x(QS::POSZ) >= 30.0 ||
      quad_state_.x(QS::POSX) <= -30.0 || quad_state_.x(QS::POSX) >= 30.0 ||
      quad_state_.x(QS::POSY) <= -30.0 || quad_state_.x(QS::POSY) >= 30.0) {
    reward = -0.02;
    return true;
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
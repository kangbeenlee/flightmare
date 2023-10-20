#include "flightlib/envs/quadrotor_env/target_quadrotor_env.hpp"

namespace flightlib {

TargetQuadrotorEnv::TargetQuadrotorEnv() : TargetQuadrotorEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/target_quadrotor_env.yaml")) {}
TargetQuadrotorEnv::TargetQuadrotorEnv(const std::string &cfg_path) : EnvBase()
{
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);
  //
  target_ptr_ = std::make_shared<TargetQuadrotor>();
  // Define quadrotor type
  target_ptr_->setType(0);
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  target_ptr_->updateDynamics(dynamics);

  // Minimum snap trajectory
  trajectory_ = MinimumSnapTrajectory();

  // define a bounding box
  world_box_ << -100, 100, -100, 100, 0, 100;
  if (!target_ptr_->setWorldBox(world_box_))
  {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = targetquadenv::kNObs;
  act_dim_ = targetquadenv::kNAct;

  // load parameters
  loadParam(cfg_);
}

TargetQuadrotorEnv::~TargetQuadrotorEnv() {}

bool TargetQuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();

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
  // obtain observations
  getObs(obs);
  return true;
}

bool TargetQuadrotorEnv::reset(Ref<Vector<>> obs, Ref<Vector<>> position, const MinimumSnapTrajectory& trajectory)
{
  quad_state_.setZero();
  quad_state_.x(QS::POSX) = position[0];
  quad_state_.x(QS::POSY) = position[1];
  quad_state_.x(QS::POSZ) = position[2];

  // Store init position
  init_position_ = position;

  // Reset quadrotor with random states
  target_ptr_->reset(quad_state_);

  // Reset control command for minimumsnap trajectory
  sim_time_ = 0.0;
  trajectory_ = trajectory;

  // Obtain observations
  getObs(obs);
  return true;
}

bool TargetQuadrotorEnv::getObs(Ref<Vector<>> obs) {
  target_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  obs.segment<targetquadenv::kNObs>(targetquadenv::kObs) = quad_obs_;
  return true;
}

Scalar TargetQuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs)
{
  // Reward function of tracker quadrotor
  Scalar total_reward = 0.0;
  return total_reward;
}

Scalar TargetQuadrotorEnv::targetStep(Ref<Vector<>> obs)
{
  Eigen::VectorXf desPosVelAcc = trajectory_.getDesiredPosVelAcc(sim_time_);
  // std::cout << ">>> sim time: " << sim_time_ << std::endl;
  // std::cout << ">>> desPosVelAcc: " << desPosVelAcc[0] << ", " << desPosVelAcc[1] << ", " << desPosVelAcc[2] << std::endl;
  // quad_state_.x[QS::POSX] = desPosVelAcc[0] + init_position_[0];
  // quad_state_.x[QS::POSY] = desPosVelAcc[1] + init_position_[1];
  quad_state_.x[QS::POSX] = desPosVelAcc[0];
  quad_state_.x[QS::POSY] = desPosVelAcc[1];  
  quad_state_.x[QS::POSZ] = desPosVelAcc[2];
  target_ptr_->setState(quad_state_);

  //
  sim_time_ += sim_dt_;

  // update observations
  getObs(obs);

  // Reward function of tracker quadrotor
  Scalar total_reward = 0.0;

  return total_reward;
}

Vector<3> TargetQuadrotorEnv::getPosition() const {
  return quad_state_.p;
}

bool TargetQuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -0.02;
    return true;
  }
  reward = 0.0;
  return false;
}

bool TargetQuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

void TargetQuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addTarget(target_ptr_);
}

std::ostream &operator<<(std::ostream &os, const TargetQuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Target Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n" << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
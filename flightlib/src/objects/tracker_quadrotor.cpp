#include "flightlib/objects/tracker_quadrotor.hpp"

namespace flightlib {

TrackerQuadrotor::TrackerQuadrotor(const std::string &cfg_path)
  : world_box_((Matrix<3, 2>() << -100, 100, -100, 100, -100, 100).finished()),
    size_(1.0, 1.0, 1.0),
    collision_(false)
{
  //
  YAML::Node cfg = YAML::LoadFile(cfg_path);

  // create quadrotor dynamics and update the parameters
  dynamics_.updateParams(cfg);

  // Default quadrotor type is tracker
  type_ = 1;

  init();
}

TrackerQuadrotor::TrackerQuadrotor(const QuadrotorDynamics &dynamics)
  : world_box_((Matrix<3, 2>() << -100, 100, -100, 100, -100, 100).finished()),
    dynamics_(dynamics), size_(1.0, 1.0, 1.0), collision_(false)
{
  // Default quadrotor type is tracker
  type_ = 1;

  init();
}

TrackerQuadrotor::~TrackerQuadrotor() {}

bool TrackerQuadrotor::run(const Command &cmd, const Scalar ctl_dt) {
  if (!setCommand(cmd)) return false;
  return run(ctl_dt);
}

bool TrackerQuadrotor::run(const Scalar ctl_dt) {
  if (!state_.valid()) return false;
  if (!cmd_.valid()) return false;

  QuadState old_state = state_;
  QuadState next_state = state_;

  // time
  const Scalar max_dt = integrator_ptr_->dtMax();
  Scalar remain_ctl_dt = ctl_dt;

  // simulation loop
  while (remain_ctl_dt > 0.0) {
    const Scalar sim_dt = std::min(remain_ctl_dt, max_dt);

    // const Vector<4> motor_thrusts_des = cmd_.isSingleRotorThrusts() ? cmd_.thrusts : runFlightCtl(sim_dt, state_.w, cmd_);
    const Vector<4> motor_thrusts_des = runFlightCtl(sim_dt, state_.w, cmd_);

    runMotors(sim_dt, motor_thrusts_des);
    // motor_thrusts_ = cmd_.thrusts;

    const Vector<4> force_torques = B_allocation_ * motor_thrusts_;

    // Compute linear acceleration and body torque
    const Vector<3> force(0.0, 0.0, force_torques[0]);
    state_.a = state_.q() * force * 1.0 / dynamics_.getMass() + gz_;

    // compute body torque
    state_.tau = force_torques.segment<3>(1);

    // dynamics integration
    integrator_ptr_->step(state_.x, sim_dt, next_state.x);

    // update state and sim time
    state_.qx /= state_.qx.norm();

    //
    state_.x = next_state.x;
    remain_ctl_dt -= sim_dt;
  }
  state_.t += ctl_dt;
  //
  constrainInWorldBox(old_state);
  return true;
}

void TrackerQuadrotor::init(void) {
  // reset
  updateDynamics(dynamics_);
  reset();
}

bool TrackerQuadrotor::reset(void) {
  state_.setZero();
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  return true;
}

bool TrackerQuadrotor::reset(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  motor_omega_.setZero();
  motor_thrusts_.setZero();
  return true;
}

void TrackerQuadrotor::clampTrustAndTorque(Vector<4>& thrust_and_torque){
    if (thrust_and_torque[0] >= T_max_)
        thrust_and_torque[0] = T_max_;
    else if (thrust_and_torque[0] <= T_min_)
        thrust_and_torque[0] = T_min_;

    if (thrust_and_torque[1] >= Mxy_max_)
        thrust_and_torque[1] = Mxy_max_;
    else if (thrust_and_torque[1] <= Mxy_min_)
        thrust_and_torque[1] = Mxy_min_;
    
    if (thrust_and_torque[2] >= Mxy_max_)
        thrust_and_torque[2] = Mxy_max_;
    else if (thrust_and_torque[2] <= Mxy_min_)
        thrust_and_torque[2] = Mxy_min_;
    
    if (thrust_and_torque[3] >= Mz_max_)
        thrust_and_torque[3] = Mz_max_;
    else if (thrust_and_torque[3] <= Mz_min_)
        thrust_and_torque[3] = Mz_min_;
}

Vector<4> TrackerQuadrotor::runFlightCtl(const Scalar sim_dt, const Vector<3> &omega, const Command &command) {
  const Scalar force = dynamics_.getMass() * command.collective_thrust;

  const Vector<3> omega_err = command.omega - omega;

  const Vector<3> body_torque_des = dynamics_.getJ() * Kinv_ang_vel_tau_ * omega_err +  state_.w.cross(dynamics_.getJ() * state_.w);

  Vector<4> thrust_and_torque(force, body_torque_des.x(), body_torque_des.y(), body_torque_des.z());
  clampTrustAndTorque(thrust_and_torque);

  const Vector<4> motor_thrusts_des = B_allocation_inv_ * thrust_and_torque;

  return dynamics_.clampThrust(motor_thrusts_des);
}

void TrackerQuadrotor::runMotors(const Scalar sim_dt, const Vector<4> &motor_thruts_des) {
  const Vector<4> motor_omega_des = dynamics_.motorThrustToOmega(motor_thruts_des);
  const Vector<4> motor_omega_clamped = dynamics_.clampMotorOmega(motor_omega_des);

  // simulate motors as a first-order system
  const Scalar c = std::exp(-sim_dt * dynamics_.getMotorTauInv());
  motor_omega_ = c * motor_omega_ + (1.0 - c) * motor_omega_clamped;

  motor_thrusts_ = dynamics_.motorOmegaToThrust(motor_omega_);
  motor_thrusts_ = dynamics_.clampThrust(motor_thrusts_);
}

bool TrackerQuadrotor::setCommand(const Command &cmd) {
  if (!cmd.valid()) return false;
  cmd_ = cmd;

  if (std::isfinite(cmd_.collective_thrust))
    cmd_.collective_thrust = dynamics_.clampThrust(cmd_.collective_thrust);

  if (cmd_.omega.allFinite())
    cmd_.omega = dynamics_.clampBodyrates(cmd_.omega);

  if (cmd_.thrusts.allFinite())
    cmd_.thrusts = dynamics_.clampThrust(cmd_.thrusts);

  // if (cmd_.velocity.allFinite())
  //   cmd_.velocity = dynamics_.clampVelocity(cmd_.velocity);

  return true;
}

bool TrackerQuadrotor::setState(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}

bool TrackerQuadrotor::setWorldBox(const Ref<Matrix<3, 2>> box) {
  if (box(0, 0) >= box(0, 1) || box(1, 0) >= box(1, 1) ||
      box(2, 0) >= box(2, 1)) {
    return false;
  }
  world_box_ = box;
  return true;
}

bool TrackerQuadrotor::constrainInWorldBox(const QuadState &old_state) {
  if (!old_state.valid()) return false;

  // violate world box constraint in the x-axis
  if (state_.x(QS::POSX) < world_box_(0, 0) ||
      state_.x(QS::POSX) > world_box_(0, 1)) {
    state_.x(QS::POSX) = old_state.x(QS::POSX);
    state_.x(QS::VELX) = 0.0;
  }

  // violate world box constraint in the y-axis
  if (state_.x(QS::POSY) < world_box_(1, 0) ||
      state_.x(QS::POSY) > world_box_(1, 1)) {
    state_.x(QS::POSY) = old_state.x(QS::POSY);
    state_.x(QS::VELY) = 0.0;
  }

  // violate world box constraint in the x-axis
  if (state_.x(QS::POSZ) <= world_box_(2, 0) ||
      state_.x(QS::POSZ) > world_box_(2, 1)) {
    //
    state_.x(QS::POSZ) = world_box_(2, 0);

    // reset velocity to zero
    state_.x(QS::VELX) = 0.0;
    state_.x(QS::VELY) = 0.0;

    // reset acceleration to zero
    state_.a << 0.0, 0.0, 0.0;
    // reset angular velocity to zero
    state_.w << 0.0, 0.0, 0.0;
  }
  return true;
}

Vector<3> TrackerQuadrotor::quaternionToEuler(QuadState& state) const {
  // From quadternion to euler
  Scalar e0 = state_.x(QS::ATTW);
  Scalar e1 = state_.x(QS::ATTX);
  Scalar e2 = state_.x(QS::ATTY);
  Scalar e3 = state_.x(QS::ATTZ);
  
  return Vector<3>(atan2(2*(e0*e1 + e2*e3), pow(e0, 2) + pow(e3, 2) - pow(e1, 2) - pow(e2, 2)),
                    asin(2*(e0*e2 - e1*e3)),
                    atan2(2*(e0*e3 + e1*e2), pow(e0, 2) + pow(e1, 2) - pow(e2, 2) - pow(e3, 2)));
}

bool TrackerQuadrotor::getState(QuadState *const state) const {
  if (!state_.valid()) return false;

  *state = state_;
  return true;
}

bool TrackerQuadrotor::getMotorThrusts(Ref<Vector<4>> motor_thrusts) const {
  motor_thrusts = motor_thrusts_;
  return true;
}

bool TrackerQuadrotor::getMotorOmega(Ref<Vector<4>> motor_omega) const {
  motor_omega = motor_omega_;
  return true;
}

bool TrackerQuadrotor::getDynamics(QuadrotorDynamics *const dynamics) const {
  if (!dynamics_.valid()) return false;
  *dynamics = dynamics_;
  return true;
}

const QuadrotorDynamics &TrackerQuadrotor::getDynamics() { return dynamics_; }

bool TrackerQuadrotor::updateDynamics(const QuadrotorDynamics &dynamics) {
  if (!dynamics.valid()) {
    std::cout << "[Quadrotor] dynamics is not valid!" << std::endl;
    return false;
  }
  dynamics_ = dynamics;
  integrator_ptr_ = std::make_unique<IntegratorRK4>(dynamics_.getDynamicsFunction(), 2.5e-3);

  B_allocation_ = dynamics_.getAllocationMatrix();
  B_allocation_inv_ = B_allocation_.inverse();

  return true;
}

bool TrackerQuadrotor::addRGBCamera(std::shared_ptr<RGBCamera> camera) {
  rgb_cameras_.push_back(camera);
  return true;
}

Vector<3> TrackerQuadrotor::getSize(void) const { return size_; }

Vector<3> TrackerQuadrotor::getPosition(void) const { return state_.p; }

std::vector<std::shared_ptr<RGBCamera>> TrackerQuadrotor::getCameras(void) const {
  return rgb_cameras_;
};

bool TrackerQuadrotor::getCamera(const size_t cam_id, std::shared_ptr<RGBCamera> camera) const {
  if (cam_id <= rgb_cameras_.size()) {
    return false;
  }

  camera = rgb_cameras_[cam_id];
  return true;
}

bool TrackerQuadrotor::getCollision() const { return collision_; }

}  // namespace flightlib
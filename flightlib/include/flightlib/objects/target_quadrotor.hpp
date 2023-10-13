#pragma once

#include <stdlib.h>
#include <csignal>


// flightlib
#include "flightlib/common/command.hpp"
#include "flightlib/common/integrator_rk4.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/dynamics/quadrotor_dynamics.hpp"
#include "flightlib/objects/object_base.hpp"
#include "flightlib/sensors/imu.hpp"
#include "flightlib/sensors/rgb_camera.hpp"
#include "flightlib/controller/velocity_controller.hpp"
#include "flightlib/data/controller_save.hpp"


namespace flightlib {

class TargetQuadrotor : ObjectBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TargetQuadrotor(const std::string& cfg_path);
  TargetQuadrotor(const QuadrotorDynamics& dynamics = QuadrotorDynamics(1.0, 0.25));
  ~TargetQuadrotor();

  // reset
  bool reset(void) override;
  bool reset(const QuadState& state);
  void init(void);

  // run the quadrotor
  bool run(const Scalar dt) override;
  bool run(const Command& cmd, const Scalar dt);

  // public get functions
  bool getState(QuadState* const state) const;
  bool getMotorThrusts(Ref<Vector<4>> motor_thrusts) const;
  bool getMotorOmega(Ref<Vector<4>> motor_omega) const;
  bool getDynamics(QuadrotorDynamics* const dynamics) const;

  const QuadrotorDynamics& getDynamics();
  Vector<3> getSize(void) const;
  Vector<3> getPosition(void) const;
  Quaternion getQuaternion(void) const;
  std::vector<std::shared_ptr<RGBCamera>> getCameras(void) const;
  bool getCamera(const size_t cam_id, std::shared_ptr<RGBCamera> camera) const;
  bool getCollision() const;

  // public set functions
  bool setState(const QuadState& state);
  bool setCommand(const Command& cmd);
  bool updateDynamics(const QuadrotorDynamics& dynamics);
  bool addRGBCamera(std::shared_ptr<RGBCamera> camera);

  // low-level controller
  Vector<4> runFlightCtl(const Scalar sim_dt, const Vector<3>& omega, const Command& cmd);

  // simulate motors
  void runMotors(const Scalar sim_dt, const Vector<4>& motor_thrust_des);

  // constrain world box
  bool setWorldBox(const Ref<Matrix<3, 2>> box);
  bool constrainInWorldBox(const QuadState& old_state);

  // PID controller function
  void setPIDControllerGain(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
                            const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
                            const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
                            const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz);

  Vector<3> quaternionToEuler(QuadState& state) const;

  //
  inline Scalar getMass(void) { return dynamics_.getMass(); };
  inline int getType(void) { return type_; };
  inline void setType(const int type) { type_ = type; };
  inline void setSize(const Ref<Vector<3>> size) { size_ = size; };
  inline void setCollision(const bool collision) { collision_ = collision; };

 private:
  // Quadrotor type: target (0) or tracker (1)
  int type_;

  // quadrotor dynamics, integrators
  QuadrotorDynamics dynamics_;
  IMU imu_;
  std::unique_ptr<IntegratorRK4> integrator_ptr_;
  std::vector<std::shared_ptr<RGBCamera>> rgb_cameras_;

  // quad control command
  Command cmd_;

  // quad state
  QuadState state_;
  Vector<3> size_;
  bool collision_;

  // PID controller for reinforcement learning action
  VelocityController velocity_controller_;
  bool save_flag_{true};

  // PID controller gain
  Scalar kp_vxy_{1.0}, ki_vxy_{0.0}, kd_vxy_{0.0},
         kp_vz_{1.0}, ki_vz_{0.0}, kd_vz_{0.0},
         kp_angle_{1.0}, ki_angle_{0.0}, kd_angle_{0.0},
         kp_wz_{1.0}, ki_wz_{0.0}, kd_wz_{0.0};

  // Clamp control input
  const Scalar thrust_max_{22.4449}, thrust_min_{0.0};
  const Scalar torque_max_{7.9355}, torque_min_{0.0};

  // Save controller output
  ControllerSave controller_save_;

  // auxiliar variablers
  Vector<4> motor_omega_;
  Vector<4> motor_thrusts_;
  Matrix<4, 4> B_allocation_;
  Matrix<4, 4> B_allocation_inv_;

  // P gain for body-rate control
  const Matrix<3, 3> Kinv_ang_vel_tau_ = Vector<3>(16.6, 16.6, 5.0).asDiagonal();
  
  // gravity
  const Vector<3> gz_{0.0, 0.0, Gz};

  // auxiliary variables
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib

#pragma once

#include <stdlib.h>

// flightlib
#include "flightlib/common/types.hpp"

namespace flightlib {

class VelocityController {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VelocityController();
  VelocityController(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
                const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
                const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
                const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz);
  ~VelocityController();

  // Control function
  Scalar controlVx(const Scalar des_vx, const Scalar vx);
  Scalar controlVy(const Scalar des_vy, const Scalar vy);
  Scalar controlVz(const Scalar des_vz, const Scalar vz, const Scalar theta, const Scalar phi);
  Scalar controlTheta(const Scalar des_theta, const Scalar theta);
  Scalar controlPhi(const Scalar des_phi, const Scalar phi);
  Scalar controlWz(const Scalar des_wz, const Scalar wz); // body rate (angular vel. w.r.t. z-axis in body frame)

  Vector<4> control(const Scalar des_vx, const Scalar des_vy, const Scalar des_vz, const Scalar des_wz,
                    const Scalar vx, const Scalar vy, const Scalar vz, const Scalar wz,
                    const Scalar phi, const Scalar theta, const Scalar psi);

  void limitControlAngle(Scalar& angle);
  void clampControlInput(Vector<4>& control_input);

  inline void setDt(const Scalar dt) { dt_ = dt; };

  void setPIDGain(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
                  const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
                  const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
                  const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz);
  
  void setQuadrotorMass(const Scalar mass);
  void setGravity(const Scalar G);

  inline const Scalar getControlTheta() { return theta_c; }
  inline const Scalar getControlPhi() { return phi_c; }

  void reset();

 private:
  // PID controller gain
  Scalar kp_vxy_{1.0}, ki_vxy_{0.0}, kd_vxy_{0.0},
         kp_vz_{1.0}, ki_vz_{0.0}, kd_vz_{0.0},
         kp_angle_{1.0}, ki_angle_{0.0}, kd_angle_{0.0},
         kp_wz_{1.0}, ki_wz_{0.0}, kd_wz_{0.0};

  // PID controller period
  Scalar dt_;

  // Control angle
  Scalar theta_c{0.0}, phi_c{0.0};

  // PID controller error
  Scalar prev_error_vx_{0.0}, prev_error_vy_{0.0}, prev_error_vz_{0.0}, prev_error_theta_{0.0}, prev_error_phi_{0.0}, prev_error_wz_{0.0};

  // PID controller integral
  Scalar integral_vx_{0.0}, integral_vy_{0.0}, integral_vz_{0.0}, integral_theta_{0.0}, integral_phi_{0.0}, integral_wz_{0.0};

  // Quadrotor parameters
  Scalar mass_{1.0}, G_{9.81};

  // Quadrotor constraints
  Scalar T_max_{44.8899}, T_min_{0.0};
  Scalar Mxy_max_{3.96774}, Mxy_min_{-3.96774};
  Scalar Mz_max_{0.15696}, Mz_min_{-0.15696};
};

}  // namespace flightlib
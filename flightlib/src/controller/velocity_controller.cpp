#include "flightlib/controller/velocity_controller.hpp"
#include <iostream>

namespace flightlib {

VelocityController::VelocityController() {}
VelocityController::VelocityController(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
                             const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
                             const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
                             const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz)
  : kp_vxy_(kp_vxy), ki_vxy_(ki_vxy), kd_vxy_(kd_vxy), kp_vz_(kp_vz), ki_vz_(ki_vz), kd_vz_(kd_vz),
    kp_angle_(kp_angle), ki_angle_(ki_angle), kd_angle_(kd_angle), kp_wz_(kp_wz), ki_wz_(ki_wz), kd_wz_(kd_wz) {}

VelocityController::~VelocityController() {}

Scalar VelocityController::controlVx(const Scalar des_vx, const Scalar vx) {
    Scalar error = des_vx - vx;
    integral_vx_ += prev_error_vx_ * dt_;
    Scalar derivative = (error - prev_error_vx_) / dt_;
    // Linear acceleration (x'')
    Scalar x_ddot = kp_vxy_ * error + ki_vxy_ * integral_vx_ + kd_vxy_ * derivative;
    prev_error_vx_ = error;
    return x_ddot;
}

Scalar VelocityController::controlVy(const Scalar des_vy, const Scalar vy) {
    Scalar error = des_vy - vy;
    integral_vy_ += prev_error_vy_ * dt_;
    Scalar derivative = (error - prev_error_vy_) / dt_;
    // Linear acceleration (y'')
    Scalar y_ddot = kp_vxy_ * error + ki_vxy_ * integral_vy_ + kd_vxy_ * derivative;
    prev_error_vy_ = error;
    return y_ddot;
}

Scalar VelocityController::controlVz(const Scalar des_vz, const Scalar vz, const Scalar theta, const Scalar phi) {
    Scalar error = des_vz - vz;
    integral_vz_ += prev_error_vz_ * dt_;
    Scalar derivative = (error - prev_error_vz_) / dt_;
    // Control thrust
    Scalar T = mass_ / (cos(phi)*cos(theta) + 1e-6) * (kp_vz_ * error + ki_vz_ * integral_vz_ + kd_vz_ * derivative + G_);
    prev_error_vz_ = error;
    return T;
}

Scalar VelocityController::controlTheta(const Scalar des_theta, const Scalar theta) {
    Scalar error = des_theta - theta;
    integral_theta_ += prev_error_theta_ * dt_;
    Scalar derivative = (error - prev_error_theta_) / dt_;
    // Control torque y
    Scalar My = kp_angle_ * error + ki_angle_ * integral_theta_ + kd_angle_ * derivative;
    prev_error_theta_ = error;
    return My;
}

Scalar VelocityController::controlPhi(const Scalar des_phi, const Scalar phi) {
    Scalar error = des_phi - phi;
    integral_phi_ += prev_error_phi_ * dt_;
    Scalar derivative = (error - prev_error_phi_) / dt_;
    // Control torque x
    Scalar Mx = kp_angle_ * error + ki_angle_ * integral_phi_ + kd_angle_ * derivative;
    prev_error_phi_ = error;
    return Mx;
}

Scalar VelocityController::controlWz(const Scalar des_wz, const Scalar wz) {
    Scalar error = des_wz - wz;
    integral_wz_ += prev_error_wz_ * dt_;
    Scalar derivative = (error - prev_error_wz_) / dt_;
    // Control torque z
    Scalar Mz = kp_wz_ * error + ki_wz_ * integral_wz_ + kd_wz_ * derivative;
    prev_error_wz_ = error;
    return Mz;
}

void VelocityController::setPIDGain(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
                           const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
                           const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
                           const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz) {
    kp_vxy_ = kp_vxy;
    ki_vxy_ = ki_vxy;
    kd_vxy_ = kd_vxy;
    kp_vz_ = kp_vz;
    ki_vz_ = ki_vz;
    kd_vz_ = kd_vz;
    kp_angle_ = kp_angle;
    ki_angle_ = ki_angle;
    kd_angle_ = kd_angle;
    kp_wz_ = kp_wz;
    ki_wz_ = ki_wz;
    kd_wz_ = kd_wz;
}

void VelocityController::setQuadrotorMass(const Scalar mass) {
    mass_ = mass;
}

void VelocityController::setGravity(const Scalar G) {
    G_ = G;
}

Vector<4> VelocityController::control(const Scalar des_vx, const Scalar des_vy, const Scalar des_vz, const Scalar des_wz,
                                 const Scalar vx, const Scalar vy, const Scalar vz, const Scalar wz,
                                 const Scalar phi, const Scalar theta, const Scalar psi) {
    // Controller 1: outter loop
    theta_c = controlVx(des_vx, vx);
    phi_c = -controlVy(des_vy, vy);

    limitControlAngle(theta_c);
    limitControlAngle(phi_c);

    // Controller 1: inner loop
    Scalar Mx = controlPhi(phi_c, phi);
    Scalar My = controlTheta(theta_c, theta);

    // Controller 2
    Scalar Mz = controlWz(des_wz, wz);

    // Controller 3
    Scalar T = controlVz(des_vz, vz, theta, phi);

    Vector<4> control_input = Vector<4>(T, Mx, My, Mz);
    clampControlInput(control_input);

    return control_input;
}

void VelocityController::limitControlAngle(Scalar& angle) {
    if (angle >= M_PI_4)
        angle = M_PI_4;
    else if (angle <= -M_PI_4)
        angle = -M_PI_4;
}

void VelocityController::clampControlInput(Vector<4>& control_input){
    if (control_input[0] >= T_max_)
        control_input[0] = T_max_;
    else if (control_input[0] <= T_min_)
        control_input[0] = T_min_;

    if (control_input[1] >= Mxy_max_)
        control_input[1] = Mxy_max_;
    else if (control_input[1] <= Mxy_min_)
        control_input[1] = Mxy_min_;
    
    if (control_input[2] >= Mxy_max_)
        control_input[2] = Mxy_max_;
    else if (control_input[2] <= Mxy_min_)
        control_input[2] = Mxy_min_;
    
    if (control_input[3] >= Mz_max_)
        control_input[3] = Mz_max_;
    else if (control_input[3] <= Mz_min_)
        control_input[3] = Mz_min_;
}

void VelocityController::reset() {
    // PID controller error
    prev_error_vx_ = 0.0;
    prev_error_vy_ = 0.0;
    prev_error_vz_ = 0.0;
    prev_error_theta_ = 0.0;
    prev_error_phi_ = 0.0;
    prev_error_wz_ = 0.0;

    // PID controller integral
    integral_vx_ = 0.0;
    integral_vy_ = 0.0;
    integral_vz_ = 0.0;
    integral_theta_ = 0.0;
    integral_phi_ = 0.0;
    integral_wz_ = 0.0;
}

}  // namespace flightlib

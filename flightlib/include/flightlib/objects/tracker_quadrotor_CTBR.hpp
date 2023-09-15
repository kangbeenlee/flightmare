// #pragma once

// #include <stdlib.h>
// #include <csignal>


// // flightlib
// #include "flightlib/common/command.hpp"
// #include "flightlib/common/integrator_rk4.hpp"
// #include "flightlib/common/types.hpp"
// #include "flightlib/dynamics/quadrotor_dynamics.hpp"
// #include "flightlib/objects/object_base.hpp"
// #include "flightlib/sensors/imu.hpp"
// #include "flightlib/sensors/rgb_camera.hpp"



// namespace flightlib {

// /** @brief Mass-normalized collective thrust & body rates action tracker */
// class TrackerQuadrotor : ObjectBase {
//  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   TrackerQuadrotor(const std::string& cfg_path);
//   TrackerQuadrotor(const QuadrotorDynamics& dynamics = QuadrotorDynamics(1.0, 0.25));
//   ~TrackerQuadrotor();

//   // reset
//   bool reset(void) override;
//   bool reset(const QuadState& state);
//   void init(void);

//   // run the quadrotor
//   bool run(const Scalar dt) override;
//   bool run(const Command& cmd, const Scalar dt);

//   // public get functions
//   bool getState(QuadState* const state) const;
//   bool getMotorThrusts(Ref<Vector<4>> motor_thrusts) const;
//   bool getMotorOmega(Ref<Vector<4>> motor_omega) const;
//   bool getDynamics(QuadrotorDynamics* const dynamics) const;

//   const QuadrotorDynamics& getDynamics();
//   Vector<3> getSize(void) const;
//   Vector<3> getPosition(void) const;
//   Quaternion getQuaternion(void) const;
//   std::vector<std::shared_ptr<RGBCamera>> getCameras(void) const;
//   bool getCamera(const size_t cam_id, std::shared_ptr<RGBCamera> camera) const;
//   bool getCollision() const;

//   // public set functions
//   bool setState(const QuadState& state);
//   bool setCommand(const Command& cmd);
//   bool updateDynamics(const QuadrotorDynamics& dynamics);
//   bool addRGBCamera(std::shared_ptr<RGBCamera> camera);

//   //
//   void clampTrustAndTorque(Vector<4>& thrust_and_torque);

//   // low-level controller
//   Vector<4> runFlightCtl(const Scalar sim_dt, const Vector<3>& omega, const Command& cmd);

//   // simulate motors
//   void runMotors(const Scalar sim_dt, const Vector<4>& motor_thrust_des);

//   // constrain world box
//   bool setWorldBox(const Ref<Matrix<3, 2>> box);
//   bool constrainInWorldBox(const QuadState& old_state);

//   Vector<3> quaternionToEuler(QuadState& state) const;

//   //
//   inline Scalar getMass(void) { return dynamics_.getMass(); };
//   inline int getType(void) { return type_; };
//   inline void setType(const int type) { type_ = type; };
//   inline void setSize(const Ref<Vector<3>> size) { size_ = size; };
//   inline void setCollision(const bool collision) { collision_ = collision; };

//  private:
//   // Quadrotor type: target (0) or tracker (1)
//   int type_;

//   // quadrotor dynamics, integrators
//   QuadrotorDynamics dynamics_;
//   IMU imu_;
//   std::unique_ptr<IntegratorRK4> integrator_ptr_;
//   std::vector<std::shared_ptr<RGBCamera>> rgb_cameras_;

//   // quad control command
//   Command cmd_;

//   // quad state
//   QuadState state_;
//   Vector<3> size_;
//   bool collision_;

//   // Clamp control input
//   const Scalar thrust_max_{22.4449}, thrust_min_{0.0};
//   const Scalar torque_max_{7.9355}, torque_min_{0.0};

//   // Quadrotor constraints
//   Scalar T_max_{43.416}, T_min_{0.0};
//   Scalar Mxy_max_{7.672}, Mxy_min_{-7.672};
//   Scalar Mz_max_{0.157}, Mz_min_{-0.157};

//   // auxiliar variablers
//   Vector<4> motor_omega_;
//   Vector<4> motor_thrusts_;
//   Matrix<4, 4> B_allocation_;
//   Matrix<4, 4> B_allocation_inv_;

//   // P gain for body-rate control
//   const Matrix<3, 3> Kinv_ang_vel_tau_ = Vector<3>(16.6, 16.6, 5.0).asDiagonal();
  
//   // gravity
//   const Vector<3> gz_{0.0, 0.0, Gz};

//   // auxiliary variables
//   Matrix<3, 2> world_box_;
// };

// }  // namespace flightlib

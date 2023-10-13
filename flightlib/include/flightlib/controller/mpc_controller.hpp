// #pragma once

// #include <stdlib.h>

// // flightlib
// #include "flightlib/common/types.hpp"

// namespace flightlib {

// class MPCController {
//  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//   MPCController();
//   ~MPCController();

//   void init(const Scalar Ts, const Scalar mass, const Scalar Ixx, const Scalar Iyy, const Scalar Izz);

//   Vector<12> nonlinearDynamics(const Vector<12> x, const Vector<4> u);
//   Matrix<12, 12> computeJacobianAprime(const Scalar phi, const Scalar theta, const Scalar psi,
//                                        const Scalar p, const Scalar q, const Scalar r, const Scalar U1);
//   Matrix<12, 4> computeJacobianBprime(const Scalar phi, const Scalar theta, const Scalar psi);
//   Matrix<12, 12> computeJacobianA(const Matrix<12, 12> A_prime);
//   Matrix<12, 4> computeJacobianB(const Matrix<12, 4> B_prime);
//   Matrix<12, 4> computeJacobianC(const Vector<12> x_bar, const Vector<4> u_bar, const Matrix<12, 12> A_prime, const Matrix<12, 4> B_prime);


//   void setPIDGain(const Scalar kp_vxy, const Scalar ki_vxy, const Scalar kd_vxy,
//                   const Scalar kp_vz, const Scalar ki_vz, const Scalar kd_vz,
//                   const Scalar kp_angle, const Scalar ki_angle, const Scalar kd_angle,
//                   const Scalar kp_wz, const Scalar ki_wz, const Scalar kd_wz);
  
//   void setQuadrotorMass(const Scalar mass);
//   void setGravity(const Scalar G);

// //   inline const Scalar getControlTheta() { return theta_c; }

//   void reset();

//  private:
//   // Sampling time
//   Scalar Ts_;

//   // Dynamics property
//   Scalar mass_{1.0}, G_{9.81};
//   Scalar Ixx_, Iyy_, Izz_;

//   // Quadrotor constraints
//   Scalar T_max_{43.4}, T_min_{0.0};
//   Scalar Mxy_max_{3.96774}, Mxy_min_{-3.96774};
//   Scalar Mz_max_{0.15696}, Mz_min_{-0.15696};
// };

// }  // namespace flightlib

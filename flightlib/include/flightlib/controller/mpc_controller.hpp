// #pragma once

// #include <stdlib.h>
// #include <ctime>             // just for testing run-times
// #include <acado_toolkit.hpp> // everything you really need
// // #include <acado_gnuplot.hpp> // just for convenient plotting within C++

// // flightlib
// // #include "flightlib/common/types.hpp"
// // #include "flightlib/common/quad_state.hpp"

// USING_NAMESPACE_ACADO
// namespace flightlib {

// class MPCController {
//  public:
  
//   MPCController();
//   ~MPCController();

//   void init(float Ts, uint N);
//   void reset();
//   bool runMPC(const float x_0,
//               const float y_0,
//               const float z_0,
//               const float u_0,
//               const float v_0,
//               const float w_0,
//               const float phi_0,
//               const float theta_0,
//               const float psi_0,
//               const float p_0,
//               const float q_0,
//               const float r_0,
//               std::vector<float>& sub_goal,
//               std::vector<float>& control);

//  private:
//   // ACADO
//   float Ts_;
//   uint N_;

//   // Dynamics property
//   float mass{1.0}, g{9.81};
//   float Ixx{0.5}, Iyy{0.5}, Izz{0.5};

//   // Quadrotor constraints
//   double T_max_{43.4}, T_min_{0.0};
//   double Mxy_max_{3.96774}, Mxy_min_{-3.96774};
//   double Mz_max_{0.15696}, Mz_min_{-0.15696};
// };

// }  // namespace flightlib

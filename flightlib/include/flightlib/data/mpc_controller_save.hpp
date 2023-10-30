// #pragma once

// #include <stdlib.h>
// #include <armadillo>

// // flightlib
// #include "flightlib/common/types.hpp"

// namespace flightlib {

// class MPCControllerSave {
//  public:
//   MPCControllerSave() {};
//   ~MPCControllerSave() {};

//   void store(const Scalar vx_des, const Scalar vy_des, const Scalar vz_des, const Scalar wz_des,
//              const Scalar T, const Scalar Mx, const Scalar My, const Scalar Mz,
//              const Scalar vx_o, const Scalar vy_o, const Scalar vz_o, const Scalar wz_o,
//              const Scalar phi_o, const Scalar theta_o, const Scalar psi_o,
//              const Scalar sim_dt)
//   {
//     input_des_(0, i) = vx_des;
//     input_des_(1, i) = vy_des;
//     input_des_(2, i) = vz_des;
//     input_des_(3, i) = wz_des;
//     input_c_(0, i) = T;
//     input_c_(1, i) = Mx;
//     input_c_(2, i) = My;
//     input_c_(3, i) = Mz;
//     output_(0, i) = vx_o;
//     output_(1, i) = vy_o;
//     output_(2, i) = vz_o;
//     output_(3, i) = wz_o;
//     output_(4, i) = phi_o;
//     output_(5, i) = theta_o;
//     output_(6, i) = psi_o;
//     time_[i] = t;
    
//     i++;
//     t += sim_dt;
//   };

//   void save() {
//     // Save to files (for visualization)
//     arma::mat input_des_vx = input_des_.row(0);
//     arma::mat input_des_vy = input_des_.row(1);
//     arma::mat input_des_vz = input_des_.row(2);
//     arma::mat input_des_wz = input_des_.row(3);
//     input_des_vx.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vx_des.txt", arma::raw_ascii);
//     input_des_vy.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vy_des.txt", arma::raw_ascii);
//     input_des_vz.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vz_des.txt", arma::raw_ascii);
//     input_des_wz.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/wz_des.txt", arma::raw_ascii);

//     arma::mat input_T = input_c_.row(0);
//     arma::mat input_Mx = input_c_.row(1);
//     arma::mat input_My = input_c_.row(2);
//     arma::mat input_Mz = input_c_.row(3);
//     input_T.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/thrust.txt", arma::raw_ascii);
//     input_Mx.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_x.txt", arma::raw_ascii);
//     input_My.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_y.txt", arma::raw_ascii);
//     input_Mz.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/torque_z.txt", arma::raw_ascii);

//     arma::mat output_vx = output_.row(0);
//     arma::mat output_vy = output_.row(1);
//     arma::mat output_vz = output_.row(2);
//     arma::mat output_wz = output_.row(3);
//     arma::mat output_phi = output_.row(4);
//     arma::mat output_theta = output_.row(5);
//     arma::mat output_psi = output_.row(6);
//     output_vx.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vx_o.txt", arma::raw_ascii);
//     output_vy.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vy_o.txt", arma::raw_ascii);
//     output_vz.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/vz_o.txt", arma::raw_ascii);
//     output_wz.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/wz_o.txt", arma::raw_ascii);
//     output_phi.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/phi_o.txt", arma::raw_ascii);
//     output_theta.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/theta_o.txt", arma::raw_ascii);
//     output_psi.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/psi_o.txt", arma::raw_ascii);

//     time_.save("/home/kblee/catkin_ws/src/flightmare/flightlib/include/flightlib/data/controller_output/time.txt", arma::raw_ascii);
//   }

//   bool isFull() {
//     if (i == buffer_) return true;
//     else return false;
//   }

//   void reset() {
//     i = 0;
//     t = 0.0;
//     input_des_ = arma::zeros(control_input_size_, buffer_);
//     input_c_ = arma::zeros(control_input_size_, buffer_);
//     output_ = arma::zeros(output_size_, buffer_);
//     time_ = arma::zeros(buffer_);
//   }

//  private:
//   unsigned int i = 0;
//   Scalar t = 0.0;
//   unsigned int control_input_size_ = 4; // vx, vy, vz, wz
//   unsigned int output_size_ = 7; // vx, vy, vz, wz
//   unsigned int buffer_ = 6000; // time buffer
//   arma::mat input_des_ = arma::zeros(control_input_size_, buffer_);
//   arma::mat input_c_ = arma::zeros(control_input_size_, buffer_);
//   arma::mat output_ = arma::zeros(output_size_, buffer_);
//   arma::mat time_ = arma::zeros(buffer_);
// };

// }  // namespace flightlib

// #include "flightlib/controller/mpc_controller.hpp"
// #include <iostream>

// namespace flightlib {

// MPCController::MPCController() {}
// MPCController::~MPCController() {}

// void MPCController::init(const Scalar Ts, const Scalar mass, const Scalar Ixx, const Scalar Iyy, const Scalar Izz) {}

// Vector<12> MPCController::nonlinearDynamics(const Vector<12> x, const Vector<4> u) {
//     Vector<12> xdot(12);
//     double phi = x(6), theta = x(7), psi = x(8);
//     double U1 = u(0);

//     xdot << x(3),
//             x(4),
//             x(5),
//             (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)) * U1 / mass_,
//             (-sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)) * U1 / mass_,
//             -G_ + cos(phi) * cos(theta) * U1 / mass_,
//             x(9) + sin(phi) * tan(theta) * x(10) + cos(phi) * tan(theta) * x(11),
//             cos(phi) * x(10) - sin(phi) * x(11),
//             (sin(phi) / cos(theta)) * x(10) + (cos(phi) / cos(theta)) * x(11),
//             ((Iyy_ - Izz_) / Ixx_) * x(10) * x(11) + u(1) / Ixx_,
//             ((Izz_ - Ixx_) / Iyy_) * x(9) * x(11) + u(2) / Iyy_,
//             ((Iyy_ - Ixx_) / Izz_) * x(9) * x(10) + u(3) / Izz_;

//     return xdot;
// }

// Matrix<12, 12> MPCController::computeJacobianAprime(const Scalar phi, const Scalar theta, const Scalar psi,
//                                                     const Scalar p, const Scalar q, const Scalar r, const Scalar U1) {
//     Matrix<12, 12> A_prime(12, 12);
//     A_prime <<  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
//                 0, 0, 0, 0, 0, 0, U1*(-sin(phi)*sin(theta)*cos(psi) + sin(psi)*cos(phi))/mass_, U1*cos(phi)*cos(psi)*cos(theta)/mass_, U1*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi))/mass_, 0, 0, 0,
//                 0, 0, 0, 0, 0, 0, U1*(-sin(phi)*sin(psi)*sin(theta) - cos(phi)*cos(psi))/mass_, U1*sin(psi)*cos(phi)*cos(theta)/mass_, U1*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/mass_, 0, 0, 0,
//                 0, 0, 0, 0, 0, 0, -U1*sin(phi)*cos(theta)/mass_, -U1*sin(theta)*cos(phi)/mass_, 0, 0, 0, 0,
//                 0, 0, 0, 0, 0, 0, q*cos(phi)*tan(theta) - r*sin(phi)*tan(theta), q*(tan(theta)*tan(theta) + 1)*sin(phi) + r*(tan(theta)*tan(theta) + 1)*cos(phi), 0, 1, sin(phi)*tan(theta), cos(phi)*tan(theta),
//                 0, 0, 0, 0, 0, 0, -q*sin(phi) - r*cos(phi), 0, 0, 0, cos(phi), -sin(phi),
//                 0, 0, 0, 0, 0, 0, q*cos(phi)/cos(theta) - r*sin(phi)/cos(theta), q*sin(phi)*sin(theta)/pow(cos(theta), 2) + r*sin(theta)*cos(phi)/pow(cos(theta), 2), 0, 0, sin(phi)/cos(theta), cos(phi)/cos(theta),
//                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(Iyy_ - Izz_)/Ixx_, q*(Iyy_ - Izz_)/Ixx_,
//                 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(-Ixx_ + Izz_)/Iyy_, 0, p*(-Ixx_ + Izz_)/Iyy_,
//                 0, 0, 0, 0, 0, 0, 0, 0, 0, q*(-Ixx_ + Iyy_)/Izz_, p*(-Ixx_ + Iyy_)/Izz_, 0;
//     return A_prime;
// }

// Matrix<12, 4> MPCController::computeJacobianBprime(const Scalar phi, const Scalar theta, const Scalar psi) {
//     Matrix<12, 4> B_prime(12, 4);
//     B_prime <<  0, 0, 0, 0,
//                 0, 0, 0, 0,
//                 0, 0, 0, 0,
//                 (sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))/mass_, 0, 0, 0,
//                 (-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))/mass_, 0, 0, 0,
//                 cos(phi)*cos(theta)/mass_, 0, 0, 0,
//                 0, 0, 0, 0,
//                 0, 0, 0, 0,
//                 0, 0, 0, 0,
//                 0, 1/Ixx_, 0, 0,
//                 0, 0, 1/Iyy_, 0,
//                 0, 0, 0, 1/Izz_;
//     return B_prime;
// }

// Matrix<12, 12> MPCController::computeJacobianA(const Matrix<12, 12> A_prime) {
//     Matrix<12, 12> I = Matrix<12, 12>::Identity(A_prime.rows(), A_prime.cols());
//     Matrix<12, 12> A = I + Ts_ * A_prime;
//     return A;
// }

// Matrix<12, 4> MPCController::computeJacobianB(const Matrix<12, 4> B_prime) {
//     return Ts_ * B_prime;
// }

// Matrix<12, 4> MPCController::computeJacobianC(const Vector<12> x_bar, const Vector<4> u_bar, const Matrix<12, 12> A_prime, const Matrix<12, 4> B_prime) {
//     return Ts_ * (nonlinearDynamics(x_bar, u_bar) - A_prime * x_bar - B_prime * u_bar);
// }

// }  // namespace flightlib

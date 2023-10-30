// #include "flightlib/controller/mpc_controller.hpp"
// #include <iostream>

// namespace flightlib {

// MPCController::MPCController() {}
// MPCController::~MPCController() {}

// void MPCController::init(float Ts, uint N) {
//     Ts_ = Ts;
//     N_ = N;
// }

// bool MPCController::runMPC(const float x_0,
//                            const float y_0,
//                            const float z_0,
//                            const float u_0,
//                            const float v_0,
//                            const float w_0,
//                            const float phi_0,
//                            const float theta_0,
//                            const float psi_0,
//                            const float p_0,
//                            const float q_0,
//                            const float r_0,
//                            std::vector<float>& command,
//                            std::vector<float>& control) {
//     // Define optimization control problem
//     ACADO::OCP ocp(0.0, N_ * Ts_, N_);
//     ACADO::DifferentialEquation f;
//     ACADO::DifferentialState x, y, z, u, v, w, phi, theta, psi, p, q, r;
//     ACADO::Control U1, U2, U3, U4; // Quadrotor control input: T, Mx, My, Mz


//     x.clearStaticCounters();
//     y.clearStaticCounters();
//     z.clearStaticCounters();
//     u.clearStaticCounters();
//     v.clearStaticCounters();
//     w.clearStaticCounters();
//     phi.clearStaticCounters();
//     theta.clearStaticCounters();
//     psi.clearStaticCounters();
//     p.clearStaticCounters();
//     q.clearStaticCounters();
//     r.clearStaticCounters();
//     U1.clearStaticCounters();
//     U2.clearStaticCounters();
//     U3.clearStaticCounters();
//     U4.clearStaticCounters();

//     // Quadrotor nonlinear dynamics
//     f << dot(x) == u;
//     f << dot(y) == v;
//     f << dot(z) == w;
//     f << dot(u) == (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)) * U1 / mass;
//     f << dot(v) == (-sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)) * U1 / mass;
//     f << dot(w) == -g + cos(phi) * cos(theta) * U1 / mass;
//     f << dot(phi) == p + sin(phi) * tan(theta) * q + cos(phi) * tan(theta) * r;
//     f << dot(theta) == cos(phi) * q - sin(phi) * r;
//     f << dot(psi) == (sin(phi) / cos(theta)) * q + (cos(phi) / cos(theta)) * r;
//     f << dot(p) == ((Iyy - Izz) / Ixx) * q * r + U2 / Ixx;
//     f << dot(q) == ((Izz - Ixx) / Iyy) * p * r + U3 / Iyy;
//     f << dot(r) == ((Iyy - Ixx) / Izz) * p * q + U4 / Izz;

//     // Constraints
//     ocp.subjectTo(f);

//     // ocp.subjectTo(-6.0 <= u <= 6.0);           // m/s
//     // ocp.subjectTo(-6.0 <= v <= 6.0);           // m/s
//     // ocp.subjectTo(-6.0 <= w <= 6.0);           // m/s
//     // ocp.subjectTo(-M_PI_4 <= phi <= M_PI_4);   // rad
//     // ocp.subjectTo(-M_PI_4 <= theta <= M_PI_4); // rad
//     // ocp.subjectTo(-M_PI <= p <= M_PI);         // rad/s
//     // ocp.subjectTo(-M_PI <= q <= M_PI);         // rad/s
//     // ocp.subjectTo(-M_PI <= r <= M_PI);         // rad/s

//     ocp.subjectTo(T_min_ <= U1 <= T_max_);          // rad/s
//     ocp.subjectTo(-Mxy_min_ <= U2 <= Mxy_max_);        // rad/s
//     ocp.subjectTo(-Mxy_min_ <= U3 <= Mxy_max_);        // rad/s
//     ocp.subjectTo(-Mz_min_ <= U4 <= Mz_max_);        // rad/s

//     // if (!set_initial_state_) {
//     //     // Set initial state
//     //     ocp.subjectTo( ACADO::AT_START, x ==  x_0 );
//     //     ocp.subjectTo( ACADO::AT_START, y ==  y_0 );
//     //     ocp.subjectTo( ACADO::AT_START, z ==  z_0 );
//     //     ocp.subjectTo( ACADO::AT_START, u ==  u_0 );
//     //     ocp.subjectTo( ACADO::AT_START, v ==  v_0 );
//     //     ocp.subjectTo( ACADO::AT_START, w ==  w_0 );
//     //     ocp.subjectTo( ACADO::AT_START, phi ==  phi_0 );
//     //     ocp.subjectTo( ACADO::AT_START, theta ==  theta_0 );
//     //     ocp.subjectTo( ACADO::AT_START, psi ==  psi_0 );
//     //     ocp.subjectTo( ACADO::AT_START, p ==  p_0 );
//     //     ocp.subjectTo( ACADO::AT_START, q ==  q_0 );
//     //     ocp.subjectTo( ACADO::AT_START, r ==  r_0 );
//     //     set_initial_state_ = true;
//     // }

//     // Provide the problem with an objective
//     uint action_dim = 4;
//     ACADO::DMatrix Q(action_dim, action_dim);
//     Q.setIdentity();
//     Q(0, 0) = 1.0; // u
//     Q(1, 1) = 1.0; // v
//     Q(2, 2) = 10.0; // w
//     Q(3, 3) = 1.0; // yaw (psi)

//     ACADO::DMatrix Q_N(action_dim, action_dim);
//     Q_N.setIdentity();
//     Q_N(0,0) = 1.0; // u
//     Q_N(1,1) = 1.0; // v
//     Q_N(2,2) = 10.0; // w
//     Q_N(3,3) = 1.0; // yaw (psi)

//     uint control_dim = 4;
//     ACADO::DMatrix R(control_dim, control_dim);
//     R.setIdentity();
//     R(0, 0) = 1.0; // T
//     R(1, 1) = 1.0; // M_x
//     R(2, 2) = 1.0; // M_y
//     R(3, 3) = 1.0; // M_z

//     ACADO::Function state, u_control, state_N;

//     // state << u;
//     // state << v;
//     // state << w;
//     // state << psi;

//     // state_N << u;
//     // state_N << v;
//     // state_N << w;
//     // state_N << psi;

//     state << x;
//     state << y;
//     state << z;
//     state << psi;

//     state_N << x;
//     state_N << y;
//     state_N << z;
//     state_N << psi;

//     u_control << U1;
//     u_control << U2;
//     u_control << U3;
//     u_control << U4;

//     ACADO::DVector ref(action_dim), ref_N(action_dim);
//     // ref(0) = command[0];
//     // ref(1) = command[1];
//     // ref(2) = command[2];
//     // ref(3) = command[3];

//     ref(0) = 0.0;
//     ref(1) = 0.0;
//     ref(2) = 0.0;
//     ref(3) = 0.0;

//     ref_N = ref;

//     ocp.minimizeLSQ(Q, state, ref);
//     ocp.minimizeLSQ(R, u_control);
//     ocp.minimizeLSQEndTerm(Q_N, state_N, ref_N);

//     // Define a solver for this problem
//     float mpc_rate = 10.0;                       // Hz
//     ACADO::RealTimeAlgorithm solver(ocp, 1 / mpc_rate); // args: problem, mpc_period
//     // OptimizationAlgorithm solver(ocp);
//     solver.set(ACADO::PRINT_COPYRIGHT, false);
//     solver.set(ACADO::PRINTLEVEL, 0);
//     solver.set(ACADO::INTEGRATOR_TYPE, ACADO::INT_RK45);
//     solver.set(ACADO::HESSIAN_APPROXIMATION, ACADO::GAUSS_NEWTON);  // GAUSS_NEWTON is for LSQ only, else I recommend BLOCK_BFGS_UPDATE
//     solver.set(ACADO::DISCRETIZATION_TYPE, ACADO::SINGLE_SHOOTING); // listen bro *direct* multiple-shooting is overrated

//     // Initial condition vector
//     ACADO::DVector x0(12);
//     x0.setZero();
//     x0(0) = x_0;
//     x0(1) = y_0;
//     x0(2) = z_0;
//     x0(3) = u_0;
//     x0(4) = v_0;
//     x0(5) = w_0;
//     x0(6) = phi_0;
//     x0(7) = theta_0;
//     x0(8) = psi_0;
//     x0(9) = p_0;
//     x0(10) = q_0;
//     x0(11) = r_0;

//     bool failed = false;
//     uint max_iters = 3;
//     for (uint i = 0; i < max_iters; ++i)
//     {
//         std::clock_t begin_time = std::clock();
//         failed = solver.solve(0.0, x0); // args: t_now, state_now
//         std::clock_t end_time = std::clock();

//         std::cout << "Iter: " << i << " | " << "Delay: " << 1000 * double(end_time - begin_time) / CLOCKS_PER_SEC << " ms" << std::endl;

//         if (failed) {
//             std::cout << "Solve failed! (ACADO prints the full details)" << std::endl;
//             return false;
//         }
//     }

//     if (!failed) {
//         // Get first optimal control input
//         ACADO::DVector u_values;
//         solver.getU(u_values);
//         control[0] = (float)u_values(0);
//         control[1] = (float)u_values(1);
//         control[2] = (float)u_values(2);
//         control[3] = (float)u_values(3);
//         return true;
//     }

//     return false;
// }

// }  // namespace flightlib
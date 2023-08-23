#include "flightlib/trajectory_planner/minimum_snap_trajectory.hpp"

namespace flightlib {

MinimumSnapTrajectory::MinimumSnapTrajectory() {}
MinimumSnapTrajectory::~MinimumSnapTrajectory() {}

Eigen::VectorXf MinimumSnapTrajectory::getPoly(int order, int der, float t) 
{
    Eigen::VectorXf poly(order);
    Eigen::VectorXf D(order);

    // Initialization
    // std::fill_n(poly.begin(), order, 1);
    poly.setConstant(1);

    int count = 0;
    for (int k = order-1; k >= 0; k--) {
        D[count] = k;
        count ++;
    }

    for (int i = 0; i < order; i++) {
        for (int j = 0; j < der; j++) {
            poly[i] = poly[i] * D[i];
            D[i] = D[i] - 1;
            if (D[i] == -1) {
                D[i] = 0;
            }
        }
    }

    for (int i = 0; i < order; i++) {
        poly[i] = poly[i] * pow(t, D[i]);
    }

    return poly;
};
  
Eigen::VectorXf MinimumSnapTrajectory::minTraj(Eigen::VectorXf waypoints, Eigen::VectorXf times, int order, int time_step)
{
    int n = waypoints.size() - 1;

    // Initialize A, and B matrix
    Eigen::MatrixXf A(order*n, order*n);
    Eigen::VectorXf B(order*n);

    for (int i = 0; i < order*n; i++) {
        B[i] = 0;
        for (int j = 0; j < order*n; j++) {
            A(i, j) = 0;
        }
    }

    // B matrix
    for (int i = 0; i < n; i++) {
        B[i] = waypoints[i];
        B[i + n] = waypoints[i + 1];
    }

    // Constraint 1 - Starting position for every segment
    Eigen::VectorXf poly_sp = getPoly(order, 0, 0); // polynomial at staring point
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < poly_sp.size(); j++) {
            A(i, order*i+j) = poly_sp[j];
        }
    }

    // Constraint 2 - Ending position for every segment
    for (int i = 0; i < n; i++) {
        Eigen::VectorXf poly_ep = getPoly(order, 0, times[i]); // polynomial at ending point
        for (int j = 0; j <poly_ep.size(); j++) {
            A(i+n, order*i+j) = poly_ep[j];
        }
    }

    // Constraint 3 - Starting position derivatives (up to order) are null
    int half_order = order / 2;
    for (int k = 1; k < half_order; k++) {
        Eigen::VectorXf poly_dev_sp = getPoly(order, k, 0); // polynomial derivatives at starting point
        for (int l = 0; l < order; l++) {
            A(2*n+k-1, l) = poly_dev_sp[l];
        }
    }

    // Constraint 4 - Ending position derivatives (up to order) are null
    for (int k = 1; k < half_order; k++) {
        Eigen::VectorXf poly_dev_ep = getPoly(order, k, times[time_step-1]);
        for (int l = 0; l < order; l++) {
            A(2*n+(half_order-1)+k-1, order*n-l-1) = poly_dev_ep[order-l-1];
        }
    }

    // Constant 5 - All derivatives are continuous at each waypoint transition
    for (int i = 0; i < n-1; i++) {
        for (int k = 1; k < order-1; k++) {
            Eigen::VectorXf getPoly_smooth_1 = getPoly(order, k, times[i]);
            Eigen::VectorXf getPoly_smooth_2 = -getPoly(order, k, 0);
            for (int ii = 0; ii < order; ii++) {
                A(2*n+2*(half_order-1) + i*2*(half_order-1)+k-1, i*order+ii) = getPoly_smooth_1[ii];
            }
            for (int jj = 0; jj < order; jj++) {
                A(2*n+2*(half_order-1) + i*2*(half_order-1)+k-1, i*order+order+jj) = getPoly_smooth_2[jj];
            }
        }
    }

    // solve for the coefficients
    Eigen::MatrixXf A_inv = A.inverse();

    // Coefficients
    Eigen::VectorXf coeff = A_inv * B;

    return coeff;
};

Eigen::VectorXf MinimumSnapTrajectory::posWayPointMin(float time, int time_step, Eigen::VectorXf way_point_times, int order,
                                Eigen::VectorXf cX, Eigen::VectorXf cY, Eigen::VectorXf cZ)
{
    float desPosX = 0;
    float desPosY = 0;
    float desPosZ = 0;

    float desVelX = 0;
    float desVelY = 0;
    float desVelZ = 0;

    float desAccX = 0;
    float desAccY = 0;
    float desAccZ = 0;

    int t_idx;
    for (int i = 0; i < time_step; i++) {
        if (time >= way_point_times[i] && time < way_point_times[i+1]) {
            t_idx = i;
            break;
        }
        t_idx = time_step;
    }

    // Scaled time (between 0 and duration of segment)
    float scale = (time - way_point_times[t_idx]);

    // Which coefficients to use
    int start = order * t_idx;
    int end = order * (t_idx + 1);

    // Set desired position, velocity and acceleration
    Eigen::VectorXf get_poly_0 = getPoly(order, 0, scale);
    Eigen::VectorXf get_poly_1 = getPoly(order, 1, scale);
    Eigen::VectorXf get_poly_2 = getPoly(order, 2, scale);

    Eigen::VectorXf Px = cX.segment(start, order);
    Eigen::VectorXf Py = cY.segment(start, order);
    Eigen::VectorXf Pz = cZ.segment(start, order);

    for (int i = 0; i < order; i++) {
        desPosX += Px[i] * get_poly_0[i];
        desPosY += Py[i] * get_poly_0[i];
        desPosZ += Pz[i] * get_poly_0[i];
    }

    for (int i = 0; i < order; i++) {
        desVelX += Px[i] * get_poly_1[i];
        desVelY += Py[i] * get_poly_1[i];
        desVelZ += Pz[i] * get_poly_1[i];
    }

    for (int i = 0; i < order; i++) {
        desAccX += Px[i] * get_poly_2[i];
        desAccY += Py[i] * get_poly_2[i];
        desAccZ += Pz[i] * get_poly_2[i];
    }

    Eigen::VectorXf desPosVelAcc(9);
    desPosVelAcc << desPosX, desPosY, desPosZ, desVelX, desVelY, desVelZ, desAccX, desAccY, desAccZ;

    return desPosVelAcc;
};

void MinimumSnapTrajectory::initTrajectory()
{

}

void MinimumSnapTrajectory::setMinimumSnapTrajectory(const Eigen::MatrixXf& way_points, const Eigen::VectorXf& segment_times)
{
    // The number of way points must be less than the number of segment times by 1 !!!

    Eigen::VectorXf way_points_x = way_points.col(0); // x-axis way points
    Eigen::VectorXf way_points_y = way_points.col(1); // y-axis way points
    Eigen::VectorXf way_points_z = way_points.col(2); // z-axis way points

    time_step_ = segment_times.size();
    total_time_ = segment_times.sum();

    // Scaled time (between 0 and duration of segment)
    way_point_times_ = Eigen::VectorXf(time_step_ + 1);
    way_point_times_[0] = 0;
    for (int i = 1; i < time_step_ + 1; i++) {
        way_point_times_[i] = way_point_times_[i - 1] + segment_times[i - 1];
    }

    coeff_x_ = minTraj(way_points_x, segment_times, order_, time_step_);
    coeff_y_ = minTraj(way_points_y, segment_times, order_, time_step_);
    coeff_z_ = minTraj(way_points_z, segment_times, order_, time_step_);
}

Eigen::VectorXf MinimumSnapTrajectory::getDesiredPosVelAcc(float time)
{
    if (time > total_time_)
        time -= total_time_;
    std::cout << ">>> time: " << time << std::endl;
    return posWayPointMin(time, time_step_, way_point_times_, order_, coeff_x_, coeff_y_, coeff_z_);
}

}  // namespace flightlib

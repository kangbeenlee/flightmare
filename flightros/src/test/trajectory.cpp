
// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/static_gate.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

// trajectory
#include <polynomial_trajectories/minimum_snap_trajectories.h>
#include <polynomial_trajectories/polynomial_trajectories_common.h>
#include <polynomial_trajectories/polynomial_trajectory.h>
#include <polynomial_trajectories/polynomial_trajectory_settings.h>
#include <quadrotor_common/trajectory_point.h>


using namespace flightlib;

int main(int argc, char *argv[]) {
  // initialize ROS
  ros::init(argc, argv, "gates_example");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate(50.0);

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr = UnityBridge::getInstance();
  SceneID scene_id{UnityScene::WAREHOUSE};
  bool unity_ready{false};

  // unity quadrotor
  std::shared_ptr<Quadrotor> quad_ptr = std::make_shared<Quadrotor>();
  QuadState quad_state;
  quad_state.setZero();
  quad_ptr->reset(quad_state);
  // define quadsize scale (for unity visualization only)
  Vector<3> quad_size(0.5, 0.5, 0.5);
  quad_ptr->setSize(quad_size);

  // Define path through gates
  std::vector<Eigen::Vector3d> way_points; // (x, y, z)
  way_points.push_back(Eigen::Vector3d(0, 10, 2.5));
  way_points.push_back(Eigen::Vector3d(5, 0, 2.5));
  way_points.push_back(Eigen::Vector3d(0, -10, 2.5));
  way_points.push_back(Eigen::Vector3d(-5, 0, 2.5));

  std::size_t num_waypoints = way_points.size();
  Eigen::VectorXd segment_times(num_waypoints);
//   segment_times << 10.0, 10.0, 10.0, 10.0;
  segment_times << 20.0, 20.0, 20.0, 20.0;
  Eigen::VectorXd minimization_weights(5);
  minimization_weights << 0.0, 1.0, 1.0, 1.0, 1.0;

  polynomial_trajectories::PolynomialTrajectorySettings trajectory_settings =
    polynomial_trajectories::PolynomialTrajectorySettings(way_points, minimization_weights, 7, 4);

  polynomial_trajectories::PolynomialTrajectory trajectory =
    polynomial_trajectories::minimum_snap_trajectories::generateMinimumSnapRingTrajectory(segment_times, trajectory_settings, 20.0, 20.0, 6.0);

  // Start racing
  ros::Time t0 = ros::Time::now();

  unity_bridge_ptr->addTracker(quad_ptr);
  // connect unity
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  FrameID frame_id = 0;

  std::cout << "---- flag ----" << std::endl;

  while (ros::ok() && unity_ready) {
    quadrotor_common::TrajectoryPoint desired_pose =
      polynomial_trajectories::getPointFromTrajectory(trajectory, ros::Duration((ros::Time::now() - t0)));
    //
    quad_state.x[QS::POSX] = (Scalar)desired_pose.position.x();
    quad_state.x[QS::POSY] = (Scalar)desired_pose.position.y();
    quad_state.x[QS::POSZ] = (Scalar)desired_pose.position.z();
    quad_state.x[QS::ATTW] = (Scalar)desired_pose.orientation.w();
    quad_state.x[QS::ATTX] = (Scalar)desired_pose.orientation.x();
    quad_state.x[QS::ATTY] = (Scalar)desired_pose.orientation.y();
    quad_state.x[QS::ATTZ] = (Scalar)desired_pose.orientation.z();

    auto start_time = std::chrono::high_resolution_clock::now();
    quad_ptr->setState(quad_state);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_diff = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "Time difference: " << time_diff.count() << " seconds" << std::endl;


    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();

    //
    frame_id += 1;
  }

  return 0;
}

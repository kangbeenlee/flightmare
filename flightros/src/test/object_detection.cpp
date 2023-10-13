
// ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
// #include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/tracker_quadrotor.hpp"
#include "flightlib/objects/target_quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

// trajectory
#include <polynomial_trajectories/minimum_snap_trajectories.h>
#include <polynomial_trajectories/polynomial_trajectories_common.h>
#include <polynomial_trajectories/polynomial_trajectory.h>
#include <polynomial_trajectories/polynomial_trajectory_settings.h>
#include <quadrotor_common/trajectory_point.h>

// opencv
#include <opencv2/opencv.hpp>
#include <random>
#include <fstream>

using namespace flightlib;

Vector<4> eulerToQuaternion(Vector<3> euler_zyx)
{
  Scalar cy = cos(euler_zyx[0] * 0.5);
  Scalar sy = sin(euler_zyx[0] * 0.5);
  Scalar cp = cos(euler_zyx[1] * 0.5);
  Scalar sp = sin(euler_zyx[1] * 0.5);
  Scalar cr = cos(euler_zyx[2] * 0.5);
  Scalar sr = sin(euler_zyx[2] * 0.5);

  Vector<4> quaternion(cy * cp * cr + sy * sp * sr,
                       cy * cp * sr - sy * sp * cr,
                       sy * cp * sr + cy * sp * cr,
                       sy * cp * cr - cy * sp * sr);
  return quaternion;
}

int main(int argc, char *argv[]) {
  // initialize ROS
  ros::init(argc, argv, "object_detection");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate(50.0);

  // publisher
  image_transport::Publisher rgb_pub1;

  // unity quadrotor
  std::shared_ptr<TrackerQuadrotor> quad_ptr1 = std::make_shared<TrackerQuadrotor>();
  Vector<3> quad_size1(0.5, 0.5, 0.5);
  quad_ptr1->setSize(quad_size1);
  QuadState quad_state1;

  std::shared_ptr<TrackerQuadrotor> tracker_ptr = std::make_shared<TrackerQuadrotor>();
  Vector<3> quad_size3(0.5, 0.5, 0.5);
  tracker_ptr->setSize(quad_size3);
  QuadState tracker_state;

  std::shared_ptr<TargetQuadrotor> target_ptr = std::make_shared<TargetQuadrotor>();
  Vector<3> quad_size4(0.5, 0.5, 0.5);
  target_ptr->setSize(quad_size4);
  QuadState target_state;

  // Create camera sensor
  std::shared_ptr<RGBCamera> rgb_camera1 = std::make_shared<RGBCamera>();

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr = UnityBridge::getInstance();
  SceneID scene_id{UnityScene::WAREHOUSE};
  bool unity_ready{false};

  // initialize publishers
  image_transport::ImageTransport it(pnh);
  rgb_pub1 = it.advertise("/rgb1", 10);

  // Add Camera sensors to each drone
  Vector<3> B_r_BC1(0.0, 0.0, 0.3);
  Matrix<3, 3> R_BC1 = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  std::cout << R_BC1 << std::endl;
  rgb_camera1->setFOV(45);
  rgb_camera1->setWidth(640);
  rgb_camera1->setHeight(640);
  rgb_camera1->setRelPose(B_r_BC1, R_BC1);
  rgb_camera1->setPostProcesscing(std::vector<bool>{true, false, false});  // depth, segmentation, optical flow
  quad_ptr1->addRGBCamera(rgb_camera1);

  // Initialization
  quad_state1.setZero();
  quad_state1.x[QS::POSX] = 0.0;
  quad_state1.x[QS::POSY] = 0.0;
  quad_state1.x[QS::POSZ] = 5.0;
  quad_ptr1->reset(quad_state1);

  tracker_state.setZero();
  tracker_state.x[QS::POSX] = -0.25;
  tracker_state.x[QS::POSY] = 3.0; 
  tracker_state.x[QS::POSZ] = 5.0; 
  tracker_ptr->reset(tracker_state);

  target_state.setZero();
  target_state.x[QS::POSX] = 0.25;
  target_state.x[QS::POSY] = 3.0; 
  target_state.x[QS::POSZ] = 5.0; 
  target_ptr->reset(target_state);

  // Connect unity
  unity_bridge_ptr->addTracker(quad_ptr1);
  unity_bridge_ptr->addTracker(tracker_ptr);
  unity_bridge_ptr->addTarget(target_ptr);
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  //
  std::string dir = "/home/kblee/catkin_ws/src/flightmare/flightros/src/test";
  std::string image, image_with_box, coordinates;

  //
  std::normal_distribution<Scalar> norm_dist_position{0.0, 1.5};
  std::normal_distribution<Scalar> norm_dist_attitude{0.0, 0.5};
  std::random_device rd;
  std::mt19937 random_gen{rd()};
  
  //
  Vector<3> euler_tracker(0, 0, 0);
  Vector<3> euler_target(0, 0, 0);

  FrameID frame_id = 1;
  while (ros::ok() && unity_ready) {
    if (frame_id % 2 == 0)
    {
      tracker_state.x[QS::POSX] = 0 + norm_dist_position(random_gen);
      tracker_state.x[QS::POSY] = 5 + norm_dist_position(random_gen);
      tracker_state.x[QS::POSZ] = 5 + norm_dist_position(random_gen);
      if (tracker_state.x[QS::POSZ] < 1.0)
        tracker_state.x[QS::POSZ] = 1.0;
      else if (tracker_state.x[QS::POSZ] > 15.0)
        tracker_state.x[QS::POSZ] = 15.0;
      euler_tracker = Vector<3>(norm_dist_attitude(random_gen), norm_dist_attitude(random_gen), norm_dist_attitude(random_gen));

      target_state.x[QS::POSX] = 0 + norm_dist_position(random_gen);
      target_state.x[QS::POSY] = 5 + norm_dist_position(random_gen);
      target_state.x[QS::POSZ] = 5 + norm_dist_position(random_gen);
      if (target_state.x[QS::POSZ] < 1.0)
        target_state.x[QS::POSZ] = 1.0;
      else if (target_state.x[QS::POSZ] > 15.0)
        target_state.x[QS::POSZ] = 15.0;
      euler_target = Vector<3>(norm_dist_attitude(random_gen), norm_dist_attitude(random_gen), norm_dist_attitude(random_gen));
    }
    tracker_state.qx = eulerToQuaternion(euler_tracker);
    tracker_ptr->setState(tracker_state);
    target_state.qx = eulerToQuaternion(euler_target);
    target_ptr->setState(target_state);

    //
    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();

    cv::Mat img;
    ros::Time timestamp = ros::Time::now();
    rgb_camera1->getRGBImage(img);

    sensor_msgs::ImagePtr rgb_msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    rgb_msg1->header.stamp = timestamp;
    rgb_pub1.publish(rgb_msg1);

    frame_id += 1;
  }

  return 0;
}

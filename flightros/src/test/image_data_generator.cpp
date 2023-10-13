
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

Vector<2> project3Dto2D(const Scalar fov, const Scalar size, const QuadState& ref, Ref<Vector<3>> p_w) {
  // World to body frame
  Matrix<4, 4> T_B_W;
  T_B_W.block<3, 3>(0, 0) = ref.q().toRotationMatrix();
  T_B_W.block<3, 1>(0, 3) = ref.p;
  T_B_W.row(3) << 0.0, 0.0, 0.0, 1.0;
  Matrix<4, 4> T_W_B = T_B_W.inverse();

  // Body to camera frame
  Matrix<4, 4> T_B_C;
  T_B_C.block<3, 3>(0, 0) = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  T_B_C.block<3, 1>(0, 3) = Vector<3>(0.0, 0.0, -0.3);
  T_B_C.row(3) << 0.0, 0.0, 0.0, 1.0;

  // World to Camera frame
  Matrix<3, 4> T_W_C = (T_B_C * T_W_B).topRows<3>();

  // Camera intrinsic matrix
  Scalar f = size / (2 * tan(fov * (M_PI / 180) / 2));
  Scalar c = size / 2;
  Matrix<3, 3> K = (Matrix<3, 3>() << 0, c, f, 0, 1, 0, f, c, 0).finished();

  // Homogeneous coordinates
  Vector<4> P_W(p_w[0], p_w[1], p_w[2], 1);

  //
  Vector<3> p_c = K * T_W_C * P_W;

  //
  Vector<2> pixel(p_c[2] / p_c[1], 640 - p_c[0] / p_c[1]);

  return pixel;
}

Matrix<8, 3> compute3DBoundingBox(const QuadState& state)
{
  // Quadrotor half length (l = W/2)
  Scalar l = 0.22;
  Scalar h = 0.01;
  Scalar offset_z = 0.02;

  // Quadrotor centor position  
  Scalar x_c = state.x(QS::POSX);
  Scalar y_c = state.x(QS::POSY);
  Scalar z_c = state.x(QS::POSZ) + offset_z;

  // 8 Points before rotated
  Vector<3> pt1(x_c - l, y_c - l, z_c + h);
  Vector<3> pt2(x_c - l, y_c - l, z_c - h);
  Vector<3> pt3(x_c + l, y_c - l, z_c - h);
  Vector<3> pt4(x_c + l, y_c - l, z_c + h);
  Vector<3> pt5(x_c - l, y_c + l, z_c + h);
  Vector<3> pt6(x_c - l, y_c + l, z_c - h);
  Vector<3> pt7(x_c + l, y_c + l, z_c - h);
  Vector<3> pt8(x_c + l, y_c + l, z_c + h);

  // Rotation matrix
  Matrix<3, 3> R_B_W = state.q().toRotationMatrix();

  Matrix<3, 8> box;
  box.block<3, 1>(0,0) = R_B_W * (pt1 - state.p) + state.p;
  box.block<3, 1>(0,1) = R_B_W * (pt2 - state.p) + state.p;
  box.block<3, 1>(0,2) = R_B_W * (pt3 - state.p) + state.p;
  box.block<3, 1>(0,3) = R_B_W * (pt4 - state.p) + state.p;
  box.block<3, 1>(0,4) = R_B_W * (pt5 - state.p) + state.p;
  box.block<3, 1>(0,5) = R_B_W * (pt6 - state.p) + state.p;
  box.block<3, 1>(0,6) = R_B_W * (pt7 - state.p) + state.p;
  box.block<3, 1>(0,7) = R_B_W * (pt8 - state.p) + state.p;

  return box.transpose();
}

Matrix<4, 2> compute2DBoundingBox(const Scalar fov, const Scalar size, const QuadState& ref, Ref<Matrix<8, 3>> box_3d_tracker)
{
  Scalar min_u = 640;
  Scalar min_v = 640;
  Scalar max_u = -1;
  Scalar max_v = -1;

  for (int i = 0; i < 8; ++i)
  {
    Vector<3> row = box_3d_tracker.row(i).transpose();
    Vector<2> pixel = project3Dto2D(fov, size, ref, row);
    min_u = std::min(min_u, pixel(0));
    min_v = std::min(min_v, pixel(1));
    max_u = std::max(max_u, pixel(0));
    max_v = std::max(max_v, pixel(1));
  }
  
  Matrix<4, 2> box_2d_tracker = (Matrix<4, 2>() << min_u, min_v,
                                           max_u, min_v,
                                           min_u, max_v,
                                           max_u, max_v).finished();
  return box_2d_tracker;
}

Matrix<8, 2> computeProjected3DBoundingBox(const Scalar fov, const Scalar size, const QuadState& ref, Ref<Matrix<8, 3>> box_3d_tracker)
{
  Matrix<8, 2> projected_box_3d_tracker;
  for (int i = 0; i < 8; ++i)
  {
    Vector<3> row = box_3d_tracker.row(i).transpose();
    Vector<2> pixel = project3Dto2D(fov, size, ref, row);
    projected_box_3d_tracker(i, 0) = pixel(0);
    projected_box_3d_tracker(i, 1) = pixel(1);
  }
  return projected_box_3d_tracker;
}

bool outOfImage(Ref<Matrix<>> box)
{
  for (int i = 0; i < 4; ++i)
  {
    if (box(i, 0) < 0 || box(i, 0) >= 640 || box(i, 1) < 0 || box(i, 1) >= 640)
      return false;
  }
  return true;
}

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

Matrix<3, 3> eulerToRotation(const Ref<Vector<3>> euler_zyx) {
  Matrix<3, 3> R_x = (Matrix<3, 3>() << 1, 0, 0,
                                        0, cos(euler_zyx[2]), -sin(euler_zyx[2]),
                                        0, sin(euler_zyx[2]), cos(euler_zyx[2])).finished();

  Matrix<3, 3> R_y = (Matrix<3, 3>() << cos(euler_zyx[1]), 0, sin(euler_zyx[1]),
                                        0, 1, 0,
                                        -sin(euler_zyx[1]), 0, cos(euler_zyx[1])).finished();

  Matrix<3, 3> R_z = (Matrix<3, 3>() << cos(euler_zyx[0]), -sin(euler_zyx[0]), 0,
                                        sin(euler_zyx[0]), cos(euler_zyx[0]), 0,
                                        0, 0, 1).finished();
  // Combined rotation matrix
  Matrix<3, 3> R = R_z * R_y * R_x;
  return R;
}

int main(int argc, char *argv[]) {
  // initialize ROS
  ros::init(argc, argv, "image_data_generator");
  ros::NodeHandle nh("");
  ros::NodeHandle pnh("~");
  ros::Rate(50.0);

  // publisher
  image_transport::Publisher rgb_pub1;
  image_transport::Publisher rgb_pub2;
  image_transport::Publisher depth_pub1;
  image_transport::Publisher depth_pub2;

  // unity quadrotor
  std::shared_ptr<TrackerQuadrotor> quad_ptr1 = std::make_shared<TrackerQuadrotor>();
  Vector<3> quad_size1(0.5, 0.5, 0.5);
  quad_ptr1->setSize(quad_size1);
  QuadState quad_state1;

  std::shared_ptr<TrackerQuadrotor> quad_ptr2 = std::make_shared<TrackerQuadrotor>();
  Vector<3> quad_size2(0.5, 0.5, 0.5);
  quad_ptr2->setSize(quad_size2);
  QuadState quad_state2;

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
  std::shared_ptr<RGBCamera> rgb_camera2 = std::make_shared<RGBCamera>();

  // Flightmare(Unity3D)
  std::shared_ptr<UnityBridge> unity_bridge_ptr = UnityBridge::getInstance();
  SceneID scene_id{UnityScene::WAREHOUSE};
  bool unity_ready{false};

  // initialize publishers
  image_transport::ImageTransport it(pnh);
  rgb_pub1 = it.advertise("/rgb1", 10);
  rgb_pub2 = it.advertise("/rgb2", 10);

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

  Vector<3> B_r_BC2(0.0, 0.0, 0.3);
  // Vector<3> euler(2.0/3.0 * M_PI, 0, 0);
  // Matrix<3, 3> R_BC2 = eulerToRotation(euler);
  Matrix<3, 3> R_BC2 = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  std::cout << R_BC2 << std::endl;
  rgb_camera2->setFOV(90);
  rgb_camera2->setWidth(640);
  rgb_camera2->setHeight(640);
  rgb_camera2->setRelPose(B_r_BC2, R_BC2);
  rgb_camera2->setPostProcesscing(std::vector<bool>{true, false, false});  // depth, segmentation, optical flow
  quad_ptr2->addRGBCamera(rgb_camera2);

  // Initialization
  quad_state1.setZero();
  quad_state1.x[QS::POSX] = -0.25;
  quad_state1.x[QS::POSY] = 0.0;
  quad_state1.x[QS::POSZ] = 5.0;
  quad_ptr1->reset(quad_state1);

  quad_state2.setZero();
  quad_state2.x[QS::POSX] = 0.25;
  quad_state2.x[QS::POSY] = 0.0;
  quad_state2.x[QS::POSZ] = 5.0;
  quad_ptr2->reset(quad_state2);

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
  unity_bridge_ptr->addTracker(quad_ptr2);
  unity_bridge_ptr->addTracker(tracker_ptr);
  unity_bridge_ptr->addTarget(target_ptr);
  unity_ready = unity_bridge_ptr->connectUnity(scene_id);

  //
  std::string dir = "/home/kblee/catkin_ws/src/flightmare/flightros/src/test";
  std::string image, image_with_box, coordinates;
  unsigned int save_45 = 0;
  unsigned int save_90 = 0;

  //
  std::normal_distribution<Scalar> norm_dist_position{0.0, 1.5};
  std::normal_distribution<Scalar> norm_dist_attitude{0.0, 0.5};
  std::random_device rd;
  std::mt19937 random_gen{rd()};
  
  //
  Vector<3> euler3(0, 0, 0);
  Vector<3> euler4(0, 0, 0);

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
      euler3 = Vector<3>(norm_dist_attitude(random_gen), norm_dist_attitude(random_gen), norm_dist_attitude(random_gen));

      target_state.x[QS::POSX] = 0 + norm_dist_position(random_gen);
      target_state.x[QS::POSY] = 5 + norm_dist_position(random_gen);
      target_state.x[QS::POSZ] = 5 + norm_dist_position(random_gen);
      if (target_state.x[QS::POSZ] < 1.0)
        target_state.x[QS::POSZ] = 1.0;
      else if (target_state.x[QS::POSZ] > 15.0)
        target_state.x[QS::POSZ] = 15.0;
      euler4 = Vector<3>(norm_dist_attitude(random_gen), norm_dist_attitude(random_gen), norm_dist_attitude(random_gen));
    }
    tracker_state.qx = eulerToQuaternion(euler3);
    tracker_ptr->setState(tracker_state);
    target_state.qx = eulerToQuaternion(euler4);
    target_ptr->setState(target_state);

    //
    unity_bridge_ptr->getRender(frame_id);
    unity_bridge_ptr->handleOutput();

    cv::Mat img, img2;
    ros::Time timestamp = ros::Time::now();
    rgb_camera1->getRGBImage(img);
    rgb_camera2->getRGBImage(img2);

    if (frame_id != 1 && frame_id % 2 == 1)
    {
      // 3D bounding box
      Matrix<8, 3> box_3d_tracker = compute3DBoundingBox(tracker_state);
      Matrix<8, 3> box_3d_target = compute3DBoundingBox(target_state);
      //
      Matrix<4, 2> box_2d_tracker = compute2DBoundingBox(45, 640, quad_state1, box_3d_tracker);
      Matrix<4, 2> box_2d_target = compute2DBoundingBox(45, 640, quad_state1, box_3d_target);
      if (outOfImage(box_2d_tracker) && outOfImage(box_2d_target))
      {
        save_45++;
        //
        image = dir + "/quadrotor_image/quadrotor_45_" + std::to_string(save_45) + ".png";
        cv::imwrite(image, img);
        // tracker
        cv::Point topLeft(box_2d_tracker(0, 0), box_2d_tracker(0, 1));
        cv::Point bottomRight(box_2d_tracker(3, 0), box_2d_tracker(3, 1));
        cv::rectangle(img, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
        // target
        topLeft = cv::Point(box_2d_target(0, 0), box_2d_target(0, 1));
        bottomRight = cv::Point(box_2d_target(3, 0), box_2d_target(3, 1));
        cv::rectangle(img, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);
        // tracker
        Scalar u_c_tracker = (box_2d_tracker(3, 0) + box_2d_tracker(0, 0)) / 2;
        Scalar v_c_tracker = (box_2d_tracker(3, 1) + box_2d_tracker(0, 1)) / 2;
        Scalar width_tracker = box_2d_tracker(3, 0) - box_2d_tracker(0, 0);
        Scalar height_tracker = box_2d_tracker(3, 1) - box_2d_tracker(0, 1);
        cv::circle(img, cv::Point(u_c_tracker, v_c_tracker), 1, cv::Scalar(0, 0, 255), 2);
        // target
        Scalar u_c_target = (box_2d_target(3, 0) + box_2d_target(0, 0)) / 2;
        Scalar v_c_target = (box_2d_target(3, 1) + box_2d_target(0, 1)) / 2;
        Scalar width_target = box_2d_target(3, 0) - box_2d_target(0, 0);
        Scalar height_target = box_2d_target(3, 1) - box_2d_target(0, 1);
        cv::circle(img, cv::Point(u_c_target, v_c_target), 1, cv::Scalar(255, 0, 0), 2);
        //
        image_with_box = dir + "/quadrotor_image_with_box/quadrotor_45_" + std::to_string(save_45) + ".png";
        // cv::imshow(image_with_box, img);
        // cv::waitKey(3);
        cv::imwrite(image_with_box, img);
        //
        coordinates = dir + "/quadrotor_label/quadrotor_45_" + std::to_string(save_45) + ".txt";
        std::ofstream outputFile(coordinates);
        //
        outputFile << "0 " << u_c_tracker / 640 << " " << v_c_tracker / 640 << " " << width_tracker / 640 << " " << height_tracker / 640 << "\n";
        outputFile << "1 " << u_c_target / 640 << " " << v_c_target / 640 << " " << width_target / 640 << " " << height_target / 640;
        outputFile.close();
      }
      else
        std::cout << ">>> Out of image!" << std::endl;

      // 2D bounding box
      box_2d_tracker = compute2DBoundingBox(90, 640, quad_state2, box_3d_tracker);
      box_2d_target = compute2DBoundingBox(90, 640, quad_state2, box_3d_target);
      if (outOfImage(box_2d_tracker) && outOfImage(box_2d_target) && save_90 < 2000)
      {
        save_90++;
        //
        image = dir + "/quadrotor_image/quadrotor_90_" + std::to_string(save_90) + ".png";
        cv::imwrite(image, img2);
        // tracker
        cv::Point topLeft(box_2d_tracker(0, 0), box_2d_tracker(0, 1));
        cv::Point bottomRight(box_2d_tracker(3, 0), box_2d_tracker(3, 1));
        cv::rectangle(img2, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
        // target
        topLeft = cv::Point(box_2d_target(0, 0), box_2d_target(0, 1));
        bottomRight = cv::Point(box_2d_target(3, 0), box_2d_target(3, 1));
        cv::rectangle(img2, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);
        // tracker
        Scalar u_c_tracker = (box_2d_tracker(3, 0) + box_2d_tracker(0, 0)) / 2;
        Scalar v_c_tracker = (box_2d_tracker(3, 1) + box_2d_tracker(0, 1)) / 2;
        Scalar width_tracker = box_2d_tracker(3, 0) - box_2d_tracker(0, 0);
        Scalar height_tracker = box_2d_tracker(3, 1) - box_2d_tracker(0, 1);
        cv::circle(img, cv::Point(u_c_tracker, v_c_tracker), 1, cv::Scalar(0, 0, 255), 2);
        // target
        Scalar u_c_target = (box_2d_target(3, 0) + box_2d_target(0, 0)) / 2;
        Scalar v_c_target = (box_2d_target(3, 1) + box_2d_target(0, 1)) / 2;
        Scalar width_target = box_2d_target(3, 0) - box_2d_target(0, 0);
        Scalar height_target = box_2d_target(3, 1) - box_2d_target(0, 1);
        cv::circle(img, cv::Point(u_c_target, v_c_target), 1, cv::Scalar(255, 0, 0), 2);
        //
        image_with_box = dir + "/quadrotor_image_with_box/quadrotor_90_" + std::to_string(save_90) + ".png";
        cv::imwrite(image_with_box, img2);
        //
        coordinates = dir + "/quadrotor_label/quadrotor_90_" + std::to_string(save_90) + ".txt";
        std::ofstream outputFile2(coordinates);
        //
        outputFile2 << "0 " << u_c_tracker / 640 << " " << v_c_tracker / 640 << " " << width_tracker / 640 << " " << height_tracker / 640 << "\n";
        outputFile2 << "1 " << u_c_target / 640 << " " << v_c_target / 640 << " " << width_target / 640 << " " << height_target / 640;
        outputFile2.close();
      }
      else
        std::cout << ">>> Out of image!" << std::endl;

      std::cout << ">>> The number of saved data: " << "[45fov: " << save_45 << " ] " << "[90fov: " << save_90 << " ] "
                << "[total: " << save_45 + save_90 << " ]" << std::endl;
      if (save_45 + save_90 >= 3000)
        break;
    }
    sensor_msgs::ImagePtr rgb_msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    rgb_msg1->header.stamp = timestamp;
    rgb_pub1.publish(rgb_msg1);
    sensor_msgs::ImagePtr rgb_msg2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img2).toImageMsg();
    rgb_msg2->header.stamp = timestamp;
    rgb_pub2.publish(rgb_msg2);

    frame_id += 1;
  }

  return 0;
}

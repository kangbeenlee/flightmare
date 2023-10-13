// std libs
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

// opencv
#include <opencv2/imgproc/types_c.h>

// Include ZMQ bindings for communications with Unity.
#include <zmqpp/zmqpp.hpp>

// flightlib
#include "flightlib/bridges/unity_message_types.hpp"
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/math.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
// #include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/tracker_quadrotor.hpp"
#include "flightlib/objects/static_object.hpp"
#include "flightlib/objects/unity_camera.hpp"
#include "flightlib/sensors/rgb_camera.hpp"


using json = nlohmann::json;
using namespace flightlib;

class UnityBridge {
 public:
  // constructor & destructor
  UnityBridge();
  ~UnityBridge(){};

  // connect function
  bool connectUnity(const SceneID scene_id);
  bool disconnectUnity(void);

  // public get functions
  bool getRender(const FrameID frame_id);
  bool handleOutput();
  bool getPointCloud(PointCloudMessage_t &pointcloud_msg,
                     Scalar time_out = 600.0);

  // public set functions
  bool setScene(const SceneID &scene_id);

  // add object
  bool addTracker(std::shared_ptr<TrackerQuadrotor> quad);
  bool addCamera(std::shared_ptr<UnityCamera> unity_camera);
  bool addStaticObject(std::shared_ptr<StaticObject> static_object);

  // public auxiliary functions
  inline void setPubPort(const std::string &pub_port) { pub_port_ = pub_port; };
  inline void setSubPort(const std::string &sub_port) { sub_port_ = sub_port; };
  // create unity bridge
  static std::shared_ptr<UnityBridge> getInstance(void) {
    static std::shared_ptr<UnityBridge> bridge_ptr =
      std::make_shared<UnityBridge>();
    return bridge_ptr;
  };

 private:
  bool initializeConnections(void);

  //
  SettingsMessage_t settings_;
  PubMessage_t pub_msg_;
  Logger logger_{"UnityBridge"};

  std::vector<std::shared_ptr<TrackerQuadrotor>> unity_quadrotors_;
  std::vector<std::shared_ptr<RGBCamera>> rgb_cameras_;
  std::vector<std::shared_ptr<StaticObject>> static_objects_;

  // ZMQ variables and functions
  std::string client_address_;
  std::string pub_port_;
  std::string sub_port_;
  zmqpp::context context_;
  zmqpp::socket pub_{context_, zmqpp::socket_type::publish};
  zmqpp::socket sub_{context_, zmqpp::socket_type::subscribe};
  bool sendInitialSettings(void);
  bool handleSettings(void);

  // timing variables
  int64_t num_frames_;
  int64_t last_downloaded_utime_;
  int64_t last_download_debug_utime_;
  int64_t u_packet_latency_;

  // axuiliary variables
  const Scalar unity_connection_time_out_{60.0};
  bool unity_ready_{false};
};

int main() {
  flightlib::UnityBridge unity_bridge;

  // Add a quad to connect to Flightmare
  QuadrotorDynamics dyn = QuadrotorDynamics(1.0, 0.2);
  std::shared_ptr<TrackerQuadrotor> quad = std::make_shared<TrackerQuadrotor>(dyn);
  unity_bridge.addTracker(quad);

  if (unity_bridge.connectUnity(UnityScene::WAREHOUSE)) {
    PointCloudMessage_t pointcloud_msg;
    pointcloud_msg.path = "/home/kblee/catkin_ws/src/flightmare/flightros/src/test/point_cloud_data/";
    pointcloud_msg.file_name = "warehouse_point_cloud";

    if (unity_bridge.getPointCloud(pointcloud_msg, 30.0)) {
      std::cout << "Successfully extracted and saved point cloud data." << std::endl;
    } else {
      std::cout << "Failed to extract point cloud data." << std::endl;
    }

    // // Remove the .ply file
    // std::experimental::filesystem::remove(pointcloud_msg.path +
    //                                       pointcloud_msg.file_name + ".ply");
  } else {
    std::cout << "Failed to connect to Unity." << std::endl;
  }

  // timeout flightmare
  usleep(5 * 1e6);

  return 0;
}
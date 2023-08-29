
// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/target_tracking_env.hpp"

namespace py = pybind11;
using namespace flightlib;

PYBIND11_MODULE(flightgym, m) {
  py::class_<TargetTrackingEnv<TrackerQuadrotorEnv>>(m, "TargetTrackingEnv_v0")
    // .def(py::init<>())
    // .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def(py::init<const std::string&, const std::string&, const bool>())
    .def("reset", &TargetTrackingEnv<TrackerQuadrotorEnv>::reset)
    .def("step", &TargetTrackingEnv<TrackerQuadrotorEnv>::step)
    .def("setSeed", &TargetTrackingEnv<TrackerQuadrotorEnv>::setSeed)
    .def("close", &TargetTrackingEnv<TrackerQuadrotorEnv>::close)
    .def("isTerminalState", &TargetTrackingEnv<TrackerQuadrotorEnv>::isTerminalState)
    .def("connectUnity", &TargetTrackingEnv<TrackerQuadrotorEnv>::connectUnity)
    .def("disconnectUnity", &TargetTrackingEnv<TrackerQuadrotorEnv>::disconnectUnity)
    .def("getNumOfEnvs", &TargetTrackingEnv<TrackerQuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &TargetTrackingEnv<TrackerQuadrotorEnv>::getObsDim)
    .def("getTargetObsDim", &TargetTrackingEnv<TrackerQuadrotorEnv>::getTargetObsDim)
    .def("getActDim", &TargetTrackingEnv<TrackerQuadrotorEnv>::getActDim)
    .def("getExtraInfoNames", &TargetTrackingEnv<TrackerQuadrotorEnv>::getExtraInfoNames)
    .def("__repr__", [](const TargetTrackingEnv<TrackerQuadrotorEnv>& a) { return "Target Tracking Environment"; });
}
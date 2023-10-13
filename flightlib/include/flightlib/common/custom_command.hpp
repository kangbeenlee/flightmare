
#pragma once

#include <cmath>

#include "flightlib/common/types.hpp"

namespace flightlib {

struct CustomCommand
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CustomCommand();
  CustomCommand(const Scalar t, const Vector<4>& velocity);

  bool valid() const;

  // Time in [s]
  Scalar t{NAN};

  // Reinforcement learning action [m/s, m/s, m/s, rad/s]
  Vector<4> velocity{NAN, NAN, NAN, NAN};
};

}  // namespace flightlib
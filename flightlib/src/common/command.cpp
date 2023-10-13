#include "flightlib/common/command.hpp"


namespace flightlib {

Command::Command() {}

Command::Command(const Scalar t, const Scalar thrust, const Vector<3>& omega) : t(t), collective_thrust(thrust), omega(omega) {}

Command::Command(const Scalar t, const Vector<4>& thrusts) : t(t), thrusts(thrusts) {}

Command::Command(const Scalar t, const Vector<4>& velocity, bool velocity_command) : t(t), velocity(velocity), velocity_command(velocity_command) {}

bool Command::valid() const
{
  return std::isfinite(t) && ((std::isfinite(collective_thrust) && omega.allFinite()) ^ thrusts.allFinite() ^ velocity.allFinite());
}

bool Command::isSingleRotorThrusts() const
{
  return std::isfinite(t) && thrusts.allFinite();
}

bool Command::isRatesThrust() const
{
  return std::isfinite(t) && std::isfinite(collective_thrust) && omega.allFinite();
}

bool Command::isVelocityAction() const
{
  return std::isfinite(t) && velocity.allFinite();
}
}  // namespace flightlib
#include "flightlib/common/custom_command.hpp"


namespace flightlib
{

CustomCommand::CustomCommand() {}
CustomCommand::CustomCommand(const Scalar t, const Vector<4>& velocity) : t(t), velocity(velocity) {}

bool CustomCommand::valid() const
{
  return std::isfinite(t) && velocity.allFinite();
}

}  // namespace flightlib
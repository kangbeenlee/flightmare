#pragma once

#include <iostream>
#include <random>
#include <deque>

#include "flightlib/common/types.hpp"


namespace flightlib {

class MovingAverageFilter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MovingAverageFilter() {};
  MovingAverageFilter(int size) : size(size) {};
  ~MovingAverageFilter() {};

  void reset(void)
  {
    sum = 0.0;
    window = std::deque<Scalar>();
  };
  
  Scalar add(Scalar val)
  {
    if ((int)window.size() >= size)
    {
        sum -= window.front();
        window.pop_front();
    }
    window.push_back(val);
    sum += val;
    return sum / window.size();
  };

 private:
  int size;
  std::deque<Scalar> window;
  Scalar sum{0.0};

};

}  // namespace flightlib

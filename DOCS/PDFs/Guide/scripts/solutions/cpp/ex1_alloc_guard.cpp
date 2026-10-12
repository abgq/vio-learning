#include "support/alloc_guard.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <vector>

using Cov = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 75, 75>;

int main() {
  std::vector<int> v;
  v.reserve(4);
  vio::AllocStats::reset();
  for (int i = 0; i < 4; ++i) v.push_back(i);
  std::printf("push_back within capacity: %d allocations\n", vio::AllocStats::allocations);
  vio::AllocStats::reset();
  v.push_back(4);
  std::printf("push_back beyond capacity: %d allocations\n", vio::AllocStats::allocations);

  Cov P;
  vio::AllocStats::reset();
  P.resize(15, 15);
  P.setIdentity();
  P.resize(75, 75);
  P.setZero();
  std::printf("fixed-max resize 15 -> 75: %d allocations\n", vio::AllocStats::allocations);

  vio::AllocStats::reset();
  Eigen::MatrixXd D(10, 10);
  D.setZero();
  std::printf("dynamic MatrixXd(10, 10): %d allocations seen by the operator-new guard\n", vio::AllocStats::allocations);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
  std::printf("now with set_is_malloc_allowed(false):\n");
  std::fflush(stdout);
  Eigen::MatrixXd E(10, 10);
  E.setZero();
  std::printf("not reached? %f\n", E(0, 0));
#endif
  return 0;
}

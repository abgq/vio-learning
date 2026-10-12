#include "support/alloc_guard.hpp"
#include <Eigen/Dense>
#include <cstdio>

double sum_ref(Eigen::Ref<const Eigen::MatrixXd> m) { return m.sum(); }

int main() {
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Prm = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Ones();
  Eigen::Matrix<double, 6, 6> Pcm = Eigen::Matrix<double, 6, 6>::Ones();
  vio::AllocStats::reset();
  double s1 = sum_ref(Pcm.block(0, 0, 3, 3));
  std::printf("col-major block: sum %g, operator-new allocations %d\n", s1, vio::AllocStats::allocations);
  vio::AllocStats::reset();
  double s2 = sum_ref(Prm.block(0, 0, 3, 3));
  std::printf("row-major block: sum %g, operator-new allocations %d\n", s2, vio::AllocStats::allocations);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
  std::printf("with set_is_malloc_allowed(false):\n");
  std::fflush(stdout);
  double s3 = sum_ref(Prm.block(0, 0, 3, 3));
  std::printf("row-major block again: sum %g\n", s3);
#endif
  return 0;
}

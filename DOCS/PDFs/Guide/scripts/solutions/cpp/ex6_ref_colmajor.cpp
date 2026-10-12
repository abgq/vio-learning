// Exercise 30.6: a column-major block binds to Eigen::Ref<const MatrixXd> without a copy.
// Meaningful in the debug-sanitize build, where EIGEN_RUNTIME_NO_MALLOC lets us forbid Eigen's allocations.
#include <Eigen/Dense>
#include <cstdio>
double sum_ref(Eigen::Ref<const Eigen::MatrixXd> m) { return m.sum(); }
int main() {
  Eigen::Matrix<double, 6, 6> Pcm = Eigen::Matrix<double, 6, 6>::Ones();
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
#endif
  double s = sum_ref(Pcm.block(0, 0, 3, 3));
#ifdef EIGEN_RUNTIME_NO_MALLOC
  std::printf("col-major block with Eigen allocations forbidden: sum %g (no allocation)\n", s);
#else
  std::printf("col-major block: sum %g (build with -DEIGEN_RUNTIME_NO_MALLOC to prove that it does not allocate)\n", s);
#endif
  return 0;
}

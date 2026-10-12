#include <Eigen/Dense>
void scale(Eigen::Ref<Eigen::MatrixXd> m) { m *= 2.0; }
int main() {
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Prm = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Ones();
  scale(Prm.block(0, 0, 3, 3));
  return 0;
}

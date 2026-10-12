#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>

Eigen::Vector3d make(double s) { return Eigen::Vector3d(s, 2 * s, 3 * s); }

int main(int argc, char** argv) {
  const int which = argc > 1 ? std::atoi(argv[1]) : 1;
  Eigen::Matrix3d R = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d a(1, 2, 3), b(4, 5, 6);
  if (which == 1) {
    auto e1 = R * (a + b);
    Eigen::Vector3d x1 = e1;
    Eigen::Vector3d ref1 = R * (a + b);
    std::printf("(1) auto R*(a+b): diff %g\n", (x1 - ref1).norm());
  } else if (which == 2) {
    auto e2 = (a + b).eval().transpose();
    Eigen::RowVector3d x2 = e2;
    std::printf("(2) [%g %g %g]\n", x2(0), x2(1), x2(2));
  } else if (which == 3) {
    auto e3 = make(1.0).head<2>();
    Eigen::Vector2d x3 = e3;
    std::printf("(3) [%g %g]\n", x3(0), x3(1));
  } else if (which == 4) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    auto e4 = A * a;
    A(0, 0) = 10.0;
    Eigen::Vector3d x4 = e4;
    std::printf("(4) [%g %g %g]\n", x4(0), x4(1), x4(2));
  } else if (which == 5) {
    // dynamic-size dangling: block of a temporary MatrixXd (heap)
    auto e5 = (Eigen::MatrixXd::Ones(4, 4) * 2.0).eval().block(0, 0, 2, 2);
    Eigen::Matrix2d x5 = e5;
    std::printf("(5) [%g %g; %g %g]\n", x5(0, 0), x5(0, 1), x5(1, 0), x5(1, 1));
  }
  std::fflush(stdout);
  return 0;
}

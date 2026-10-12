#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstdlib>

constexpr int kMax = 75;
using CovFM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, kMax, kMax>;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Blk15FM = Eigen::Matrix<double, 15, Eigen::Dynamic, Eigen::ColMajor, 15, kMax>;

struct FixedMax {
  CovFM P;
  Mat15 Phi, Q, tmp15;
  Blk15FM tmpIC;
  void step() noexcept {
    const Eigen::Index nc = P.rows() - 15;
    tmp15.noalias() = Phi * P.topLeftCorner<15, 15>();
    P.topLeftCorner<15, 15>().noalias() = tmp15 * Phi.transpose();
    P.topLeftCorner<15, 15>() += Q;
    tmpIC.resize(15, nc);
    tmpIC.noalias() = Phi * P.topRightCorner(15, nc);
    P.topRightCorner(15, nc) = tmpIC;
    P.bottomLeftCorner(nc, 15) = tmpIC.transpose();
  }
};

struct DynamicNaive {
  Eigen::MatrixXd P, Phi, Q;
  void step() {
    const Eigen::Index nc = P.rows() - 15;
    P.topLeftCorner(15, 15) = Phi * P.topLeftCorner(15, 15) * Phi.transpose() + Q;
    Eigen::MatrixXd PIC = Phi * P.topRightCorner(15, nc);
    P.topRightCorner(15, nc) = PIC;
    P.bottomLeftCorner(nc, 15) = PIC.transpose();
  }
};

struct DynamicWork {
  Eigen::MatrixXd P, Phi, Q, tmp15, tmpIC;
  void step() {
    const Eigen::Index nc = P.rows() - 15;
    tmp15.noalias() = Phi * P.topLeftCorner(15, 15);
    P.topLeftCorner(15, 15).noalias() = tmp15 * Phi.transpose();
    P.topLeftCorner(15, 15) += Q;
    tmpIC.noalias() = Phi * P.topRightCorner(15, nc);
    P.topRightCorner(15, nc) = tmpIC;
    P.bottomLeftCorner(nc, 15) = tmpIC.transpose();
  }
};

template <class F>
double time_it(F& f, int steps) {
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < steps; ++k) f.step();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / steps;
}

int main(int argc, char** argv) {
  const int steps = argc > 1 ? std::atoi(argv[1]) : 10000;
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Random(kMax, kMax);
  P0 = (P0 * P0.transpose()).eval() / kMax + Eigen::MatrixXd::Identity(kMax, kMax);
  Eigen::MatrixXd W0 = Eigen::MatrixXd::Random(15, 15); Eigen::MatrixXd Phi0 = Eigen::MatrixXd::Identity(15, 15) + 0.005 * (W0 - W0.transpose());
  Eigen::MatrixXd Q0 = 1e-6 * Eigen::MatrixXd::Identity(15, 15);
  FixedMax a; a.P = P0; a.Phi = Phi0; a.Q = Q0;
  DynamicNaive b; b.P = P0; b.Phi = Phi0; b.Q = Q0;
  DynamicWork c; c.P = P0; c.Phi = Phi0; c.Q = Q0; c.tmp15.resize(15, 15); c.tmpIC.resize(15, kMax - 15);
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
  const double ta = time_it(a, steps);
  const double tc = time_it(c, steps);
  Eigen::internal::set_is_malloc_allowed(true);
  const double tb = time_it(b, steps);
#else
  const double ta = time_it(a, steps);
  const double tb = time_it(b, steps);
  const double tc = time_it(c, steps);
#endif
  std::printf("per step: fixed-max %.2f us, dynamic (temporaries) %.2f us, dynamic (workspaces) %.2f us; check %.3e %.3e\n",
              ta, tb, tc, (Eigen::MatrixXd(a.P) - b.P).norm() / b.P.norm(), (b.P - c.P).norm() / b.P.norm());
  return 0;
}

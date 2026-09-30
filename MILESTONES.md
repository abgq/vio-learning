# MILESTONES.md

## Global Exit Criteria (apply to every milestone)
- [ ] Configures, builds and tests from a clean `build/` under **both** `debug-sanitize` and `release`, using the MSYS2 CLANG64 toolchain.
- [ ] Zero compiler warnings (enforced by `-Werror`), zero Catch2 failures, zero ASan/UBSan reports (enforced by `-fno-sanitize-recover=all` and `halt_on_error`).
- [ ] Every new analytic Jacobian has a derivation in `docs/derivations.md` and a finite-difference test.
- [ ] Every new Core hot-path function is covered by an allocation-guard test.
- [ ] `PROGRESS.md` is updated: status, decisions taken, open questions.
- [ ] Milestones marked **Checkpoint review** get an external review (Reviewer Mode in `AGENTS.md`) before the next milestone starts.

---

## Milestone 1: Build Harness & Dependency Scaffolding
*   **What You Will Do:** The toolchain is already installed: MSYS2 CLANG64 with Clang 22, CMake 4.4, Ninja 1.13 and Git, all on `PATH`. Optionally add a debugger with `pacman -S mingw-w64-clang-x86_64-lldb`. Run `git init` and add a `.gitignore` for `build/` and `data/`. Create the directory hierarchy, the root `CMakeLists.txt`, `cmake/Warnings.cmake`, `cmake/Dependencies.cmake` and `CMakePresets.json`. Fetch Eigen 3.4.0 and Catch2 v3.5.2 via `FetchContent` as `SYSTEM`. Create an empty `vio_core` library and a minimal Catch2 sanity test that includes Eigen and does a trivial fixed-size matrix operation.
*   **Why You Will Do It:** Ensures deterministic library versions across machines and activates strict compiler warnings and sanitizers from the first commit, preventing silent mathematical corruption or memory bugs in fixed-size Eigen matrices.
*   **Concepts to Own:** Target-scoped vs. directory/global flags, and why warnings must not reach fetched dependencies. What `SYSTEM` includes do. `PUBLIC`/`PRIVATE`/`INTERFACE` propagation. Why UBSan passes silently without `-fno-sanitize-recover`. What `EIGEN_RUNTIME_NO_MALLOC` can and cannot catch. Why ASan on Windows needs its runtime DLL on `PATH`, and why leak detection isn't available there.
*   **Exit Criteria:**
    - [ ] `cmake --list-presets` shows `debug-sanitize` and `release`. Configure, build and test presets all exist.
    - [ ] The sanity test passes under both presets, and Eigen's headers produce zero warnings under the full warning set.
    - [ ] **Sanitizer proof (manual, one-off):** temporarily add an out-of-bounds read and then a signed-integer overflow to the sanity test. Confirm that `ctest --preset debug-sanitize` **fails** for each, then remove them. Record the result in `PROGRESS.md`.
    - [ ] `tests/support/alloc_guard.hpp` exists, with a test proving it detects a deliberate `new` and reports zero for a fixed-size Eigen product.
    - [ ] First git commit made.

## Milestone 2: 3D Lie Group Math on SO(3) — *Checkpoint review*
*   **What You Will Do:** Implement `so3.hpp`: `skew`, `Exp`/`Log` for rotation matrices, `Exp_q`/`Log_q` for quaternions, and the right Jacobian $J_r$ with its inverse $J_r^{-1}$, including small-angle Taylor branches. Build `tests/support/numerical_jacobian.hpp` on the $\boxplus/\boxminus$ operators of `AGENTS.md` C4.
*   **Why You Will Do It:** Euler angles suffer from gimbal lock. VIO stores orientation as a quaternion but estimates errors in a flat 3D tangent space, and these functions bridge the two. A single sign error here causes immediate filter divergence.
*   **Concepts to Own:** What "local/right perturbation" means physically (an error expressed in the body frame). The double cover $q \equiv -q$. Why $\mathrm{Log}$ is ill-conditioned near $\pi$. Why the Taylor branch exists (0/0), how the threshold is chosen relative to machine epsilon, and which order is needed. What $J_r$ represents.
*   **Exit Criteria:**
    - [ ] `skew(a) * b == a.cross(b)`.
    - [ ] $\mathrm{Exp}(\phi) \in SO(3)$: $\|R^TR - I\| < 10^{-12}$, $\det R = 1$.
    - [ ] $\mathrm{Log}(\mathrm{Exp}(\phi)) = \phi$ for random $\|\phi\| \in [0, \pi - \epsilon]$, **and** for $\|\phi\| \in \{0, 10^{-12}, 10^{-8}, 10^{-4}\}$, **and** near $\pi$.
    - [ ] $\mathrm{Exp}(\mathrm{Log}(R)) = R$ for random $R$, including rotation angles within $10^{-6}$ of $\pi$.
    - [ ] $\mathrm{Exp}_q(\phi)$ and $\mathrm{Exp}(\phi)$ give the same rotation, and $\mathrm{Log}_q(q) = \mathrm{Log}_q(-q)$.
    - [ ] Finite-difference check: $\mathrm{Exp}(\phi + \delta) \approx \mathrm{Exp}(\phi)\,\mathrm{Exp}(J_r(\phi)\,\delta)$, and $J_r^{-1} J_r = I$.
    - [ ] No function allocates (allocation-guard test).

## Milestone 3: EuRoC Dataset Streamer
*   **What You Will Do:** Parse `imu0/data.csv`, `state_groundtruth_estimate0/data.csv`, `imu0/sensor.yaml` (noise densities) and `cam0/sensor.yaml` (`T_BS`, intrinsics, distortion; unused until M8). Decimate the ground truth to a configurable rate (default 10 Hz) **by timestamp, not by index**. Load everything up front, then expose a time-ordered event stream (`std::variant<ImuSample, PoseMeasurement>`, extended in M8) as a two-pointer merge over the presorted arrays.
*   **Why You Will Do It:** Real sensors run on different clocks. Processing a measurement out of temporal order destroys the estimator's timeline. This builds the strict event ordering needed for real hardware.
*   **Concepts to Own:** Why an update must never be applied before the state is propagated to its timestamp. Why a merge of sorted arrays beats a `std::priority_queue` here. The EuRoC frame semantics (`p_RS_R`, `q_RS`, column order). Why the ground-truth file starts later than the IMU file.
*   **Exit Criteria:**
    - [ ] Tolerates the `#` header line, CRLF endings and trailing whitespace. A malformed row produces a clear load-time error (file and line), never UB.
    - [ ] Fixture tests (`tests/data/`): exact values of the first row, including correct quaternion w-first handling and the gyro-bias-before-accel-bias column order.
    - [ ] Stream timestamps are non-decreasing. Event count equals the sum of the inputs. **Tie rule: an IMU sample precedes a pose at an equal timestamp**, and a test enforces it.
    - [ ] The decimated ground-truth rate is within ±1 sample/s of the target.
    - [ ] Iterating the stream performs zero allocations (allocation-guard test).
    - [ ] On MH_01_easy: log the IMU rate, gaps (> 1.5× nominal Δt) and duplicate timestamps. Asserting on them is not required.
    - [ ] YAML parser decision logged.

## Milestone 4: Nominal IMU Kinematics
*   **What You Will Do:** Implement nominal propagation (`AGENTS.md` F1) using Δt from timestamps. Choose an integration scheme (e.g. Euler with zero-order hold, or midpoint), justify it and log it. Initialize from the first ground-truth sample at or after the stream start ($p, v, q, b_a, b_g$ all come from the GT columns). Build `tests/support/synthetic_trajectory.hpp` to generate noise-free IMU data from analytic trajectories.
*   **Why You Will Do It:** The drone is effectively blind between measurement updates. The IMU engine bridges this gap, forward-predicting the physical state in the milliseconds between them.
*   **Concepts to Own:** Specific force, and why an accelerometer at rest reads $+g$ upward. Why gravity is added back. The local truncation error of the chosen scheme, and why the attitude update is multiplicative.
*   **Exit Criteria (synthetic, zero noise):**
    - [ ] Stationary input ($a_m = R_{wb}^T(-g_w)$, $\omega_m = 0$): the state stays constant to $10^{-9}$ over 60 s.
    - [ ] Constant $\omega$: the orientation matches the closed form $q_0 \otimes \mathrm{Exp}_q(\omega t)$.
    - [ ] Circular trajectory with a closed-form solution: the error is bounded, and **halving Δt reduces the error by the factor the scheme predicts** (convergence-order test).
    - [ ] $\big|\|q\| - 1\big| < 10^{-12}$ after $10^6$ steps.
    - [ ] Propagation performs zero allocations.
    - [ ] Real-data smoke test: 10 s of dead reckoning on MH_01 from the GT init with GT biases produces no NaN/Inf. Report the drift, but don't assert on it.

## Milestone 5: Error Covariance Propagation — *Checkpoint review*
*   **What You Will Do:** Derive $F_c$ and $G_c$ under the frozen conventions and record them in `docs/derivations.md`. Discretize into $\Phi$ and $Q_d$ using the EuRoC continuous noise densities (`AGENTS.md` F2). Propagate the 15×15 covariance $P$ at IMU rate from a configurable $P_0$.
*   **Why You Will Do It:** The filter needs self-awareness of its own accumulating inertial drift. If uncertainty doesn't grow during flight, the Kalman gain goes to zero and the filter rejects valid measurement corrections.
*   **Concepts to Own:** Why error dynamics stay nearly linear when the nominal dynamics don't. The units of a continuous noise density, and where the $\Delta t$ in $Q_d$ comes from. Why position variance grows like $t^3$. Why individual diagonal entries of $P$ are **not** guaranteed to grow monotonically (the attitude block rotates variance between axes).
*   **Exit Criteria:**
    - [ ] **Numerical Jacobian test:** each column of $\Phi$ matches $\big(f(\hat x \boxplus \epsilon e_i) \boxminus f(\hat x)\big)/\epsilon$ for the discrete propagation $f$, on a non-trivial state (nonzero $\omega$, non-identity $q$, nonzero biases), within an $O(\Delta t^2)$ tolerance.
    - [ ] $P$ stays symmetric (to $10^{-12}$ relative) and PSD (min eigenvalue $\ge -10^{-12}\|P\|$) after $10^5$ steps.
    - [ ] Analytic check: with $P_0 = 0$, only accelerometer noise and a fixed orientation, the horizontal position variance matches $\sigma_a^2 t^3 / 3$.
    - [ ] Monte Carlo check: over ≥ 500 seeded noisy synthetic runs, the sample covariance of $x_{true} \boxminus \hat x$ agrees with the propagated $P$ (diagonal ratios within statistically justified bounds).
    - [ ] Covariance propagation performs zero allocations.

## Milestone 6: Pose Measurement Update & Error Injection
*   **What You Will Do:** Use the decimated ground-truth poses **with synthetic noise** (`AGENTS.md` C6; fixed seed, $R_z$ matching the injected noise) as a mock sensor. Compute the $\boxminus$ residual, derive $H$ (6×15) and record it, compute the gain with a Cholesky/LDLT solve, gate with χ², update with Joseph form, inject, and reset with the exact $G$ (`AGENTS.md` F3–F4).
*   **Why You Will Do It:** This is the core of the ES-EKF. Keeping the error vector near zero keeps the linearization valid. A known-noise "mock camera" proves the back end works before dealing with messy pixels.
*   **Concepts to Own:** Why the Joseph form is used, and when the simple form loses PSD. Why solve rather than invert. Where the reset Jacobian $G$ comes from and what $G \approx I$ throws away. Which states become observable from pose-only measurements, and why the biases converge.
*   **Exit Criteria:**
    - [ ] $H$ passes the finite-difference test at a non-trivial state.
    - [ ] $P_{prior} - P_{post}$ is PSD after an update.
    - [ ] After injection and reset: all quaternions are unit, and a second update with the same $z$ gives a residual consistent with the first update's correction.
    - [ ] Toy test: as $R_z \to 0$, the posterior pose converges to $z$.
    - [ ] Synthetic run with known constant biases: the bias estimates converge to within their own $3\sigma$.
    - [ ] Update, injection and reset perform zero allocations.
    - [ ] Full MH_01 run with noisy GT updates: no divergence, no NaN, and the χ² rejection count is logged.

## Milestone 7: Telemetry & Filter Consistency — *Checkpoint review (back end freeze)*
*   **What You Will Do:** Write a CSV logger that buffers in preallocated memory and flushes outside the hot loop. Log $\hat x$, $\mathrm{diag}(P)$, GT, NIS and NEES. Write `scripts/plot_consistency.py` (3σ envelopes, NEES/NIS against χ² bounds). Optionally compute ATE with `evo`.
*   **Why You Will Do It:** Proves the IMU kinematics and covariance propagation are solid, so that once vision is introduced, any drift is a vision bug and not a physics or EKF bug. Comparing the pose against the same GT it was fed is circular, so consistency is measured with **NEES/NIS** and on **states that are not directly measured** (velocity, biases).
*   **Concepts to Own:** NEES vs. NIS, and their χ² distributions. What over- and under-confidence look like in each. How to tune $Q$ and $R$ without cheating.
*   **Exit Criteria:**
    - [ ] Synthetic Monte Carlo (≥ 50 seeded runs): average NEES lies inside the two-sided 95% χ² interval for 15 DOF over most of the run.
    - [ ] NIS falls inside its per-update 95% bound for ~95% of the updates.
    - [ ] EuRoC MH_01: the fraction of velocity and bias errors inside 3σ is reported, and is ≥ 95%.
    - [ ] Logging performs zero allocations inside the streaming loop.
    - [ ] Plots committed or reproducible from a single script invocation.

## Milestone 8: Feature Detection & Optical Flow Tracking (2D Image Space)
*   **What You Will Do:** Add `cam0` images (20 Hz) to the event stream as an `ImageEvent`. Detect Shi-Tomasi corners with grid bucketing and a minimum-distance mask. Track them with pyramidal LK (`cv::calcOpticalFlowPyrLK`) plus a forward–backward consistency check. Keep persistent track IDs and replenish features below a threshold. Undistort the **tracked points** (not the whole image) into normalized coordinates using the `cam0` radial-tangential model. The frontend is developed and tested standalone; the estimator doesn't consume it yet. Pose updates remain available only as a test option.
*   **Why You Will Do It:** Isolates the 2D pixel tracking pipeline so you can verify feature persistence and uniform image distribution without also fighting 3D geometry.
*   **Concepts to Own:** Brightness constancy and the aperture problem. Why pyramids exist. Why a uniform feature distribution improves geometric conditioning. Why undistort points rather than images. Why the frontend is exempt from zero-heap rules, and what "reuse buffers" means for `cv::Mat`.
*   **Exit Criteria:**
    - [ ] Synthetic textured image warped by a known sub-pixel translation: the recovered flow is within 0.1 px.
    - [ ] The forward–backward check rejects artificially corrupted tracks.
    - [ ] Bucketing bounds the per-cell feature count, and track IDs stay stable across frames.
    - [ ] Undistortion test: distort, then undistort, a grid of points and recover them within a stated tolerance.
    - [ ] On MH_01, log features/frame, mean track length and grid coverage.
    - [ ] OpenCV is linked only to `vio_frontend`, and `vio_core` still builds without OpenCV.

## Milestone 9: Epipolar Geometry, Outlier Rejection & Track Management
*   **What You Will Do:** Build the feature lifecycle manager (NEW → TRACKED → LOST) with a fixed-capacity feature pool and per-feature observation history bounded by `kMaxClones`. Reject outliers with Essential-matrix RANSAC (`cv::findEssentialMat` on **normalized** coordinates with identity K and threshold = pixel threshold / f).
*   **Why You Will Do It:** Optical flow drifts and tracks false shadows. Feeding raw pixel drift into an EKF corrupts it. RANSAC keeps only physically possible rigid-body motion.
*   **Concepts to Own:** The epipolar constraint $x_2^T E x_1 = 0$. Why five points are the minimum. Why the RANSAC threshold must be in normalized units here. **The pure-rotation degeneracy**: what happens to $E$ at zero baseline, and how the pipeline behaves when the drone hovers.
*   **Exit Criteria:**
    - [ ] Synthetic two-view scene (random 3D points, known relative pose, 20–30% injected outliers): ≥ 95% of outliers rejected and ≥ 95% of inliers kept.
    - [ ] The pure-rotation case is tested and its behavior documented. It must not crash or reject everything silently.
    - [ ] Lifecycle transitions are unit-tested. Observation history never exceeds `kMaxClones`.
    - [ ] On MH_01: log inlier ratio per frame and track length distribution.

## Milestone 10a: Stochastic Cloning & Sliding Window
*   **What You Will Do:** Extend the state to the frozen layout $15 + 6N$ (`AGENTS.md` C3) with fixed-*max*-size Eigen storage. At each image timestamp, propagate the IMU to that exact time, clone $(p_{wb}, q_{wb})$, and augment $P$ with the cloning Jacobian $J$ (derive and record it). Propagate the cross-covariances (`AGENTS.md` F2), marginalize the oldest clone in place when the window is full, and extend injection/reset to the clones. Decide `kMaxClones` and log it.
*   **Why You Will Do It:** A landmark triangulated from past poses is correlated with those poses. Without past poses in the state, a visual update double-counts information and the filter becomes overconfident. Cloning is how the MSCKF keeps that correlation.
*   **Concepts to Own:** Why the clones must be correlated with the IMU state at creation. Why clones are not propagated. Why marginalizing a Gaussian in covariance form is just deleting rows and columns. The stack-size budget of a $(15+6N_{max})^2$ matrix, and why it lives as a member.
*   **Exit Criteria:**
    - [ ] After cloning, the new clone's covariance block and its cross-covariances equal the corresponding IMU pose blocks exactly.
    - [ ] **Invariance test:** with only pose updates active, the IMU estimate and IMU covariance with cloning enabled equal those without cloning to $10^{-10}$.
    - [ ] Marginalization matches a reference computed with a dynamic matrix (test-only allocation is allowed there).
    - [ ] $P$ stays symmetric and PSD over 1000 clone/marginalize cycles, with zero allocations.

## Milestone 10b: MSCKF Visual Update (True VIO) — *Checkpoint review*
*   **What You Will Do:** Implement `camera_model` (projection plus Jacobians with respect to the clone pose and the feature position; derive and record them) and `triangulation` (linear initialization, then Gauss–Newton over the clone poses, with depth and conditioning checks). Implement `msckf_update`: per-feature residual stacking, left-nullspace projection via Householder QR, χ² gating, capped row stacking, optional QR compression, then the F3/F4 update across the IMU and clones. Use the update triggers from `AGENTS.md` F5. Ground truth may be used **only** for the initial state, and that use is logged.
*   **Why You Will Do It:** Replaces the mock ground-truth measurements with real pixel geometry constraints, turning the inertial dead-reckoning engine into a full Visual-Inertial Odometry system.
*   **Concepts to Own:** Why the nullspace projection removes the landmark from the problem without losing information about the poses. Why at least 3 observations are needed. The 4 unobservable directions of VIO, and why the standard EKF becomes overconfident in yaw (the motivation for FEJ). Monocular scale, and why the IMU makes it observable.
*   **Exit Criteria:**
    - [ ] The projection Jacobians pass finite-difference tests with respect to the clone $\delta p$, clone $\delta\theta$ and $p_f$.
    - [ ] Triangulation: exact on noise-free synthetic data, and bounded error under pixel noise. Negative-depth and low-parallax features are rejected.
    - [ ] Nullspace: $\|A^T H_f\| < 10^{-10}$, and $A$ has orthonormal columns.
    - [ ] Full synthetic VIO (known landmarks, simulated IMU and pixels): runs without divergence, with NEES reported (yaw inconsistency is expected and documented).
    - [ ] The update path performs zero allocations with fixed-max workspaces.
    - [ ] EuRoC MH_01_easy end to end: no divergence, and ATE RMSE reported after SE(3) alignment. First-pass target < 0.5 m, stretch < 0.2 m.
*   **Stretch Goals:** First-Estimates Jacobians, online extrinsic and time-offset calibration, stereo (`cam1`), the harder sequences (V1_03, MH_05).

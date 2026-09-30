# PROGRESS.md

Shared handoff file between AI models and sessions. Read it at the start of every session; update it at the end of every milestone.

## Current Status
- **Milestone:** 1 — Build Harness & Dependency Scaffolding
- **Status:** Not started

## Decision Log
| Date | Decision | Rationale |
|---|---|---|
| 2026-09-30 | Orientation stored as $q_{wb}$ (body→world) | Matches EuRoC `q_RS`; eliminates direction ambiguity |
| 2026-09-30 | Local (right) perturbation: $q = \hat q \otimes \mathrm{Exp}(\delta\theta)$, $\delta\theta$ in body frame | Attitude error decoupled from global yaw; standard in ESKF literature (Solà) |
| 2026-09-30 | MSCKF with stochastic cloning, sliding window of `kMaxClones` | A 15-dim state cannot consistently absorb landmark-based visual updates |
| 2026-09-30 | Extrinsics and cam–IMU time offset fixed from `sensor.yaml` | Keep scope manageable; online calibration is a stretch goal |
| 2026-09-30 | Exact reset Jacobian $G$ (not $G \approx I$) | Cheap, and removes a known approximation from consistency analysis |
| 2026-09-30 | Native Windows with the MSYS2 CLANG64 toolchain (Clang 22, libc++, CMake 4.4, Ninja 1.13); no WSL | User preference for native Windows. MSYS2 ships ASan/UBSan runtimes only for CLANG64. UCRT64 GCC 16 failed to link `-lasan`/`-lubsan` (verified 2026-09-30) |
| 2026-09-30 | UCRT64 environment uninstalled; `C:\msys64\clang64\bin` on system PATH | One toolchain only; avoids mixing GCC and Clang binaries |

## Toolchain Verification (2026-09-30)
Standalone programs compiled with `clang++ -fsanitize=address,undefined -fno-sanitize-recover=all`:
- Heap out-of-bounds read: ASan reported `heap-buffer-overflow` with file:line for both the read and the allocation. Exit code 1.
- Signed integer overflow: UBSan reported `runtime error: signed integer overflow` with file:line. Exit code 1.

The Milestone 1 exit criterion still requires repeating this proof inside the CTest harness.

## Open Questions
- Integration scheme for nominal propagation (decide in M4).
- YAML: `yaml-cpp` vs. minimal hand-written parser (decide in M3).
- `kMaxClones` value (decide in M10a; default 10).

## Milestone Log
| Milestone | Status | Completed | Checkpoint review | Notes |
|---|---|---|---|---|
| 1 | Not started | | — | |
| 2 | | | Required | |
| 3 | | | — | |
| 4 | | | — | |
| 5 | | | Required | |
| 6 | | | — | |
| 7 | | | Required | |
| 8 | | | — | |
| 9 | | | — | |
| 10a | | | — | |
| 10b | | | Required | |

## Review Log
<!-- Append: date, reviewer model, milestone, verdict per exit criterion, blocking issues. -->

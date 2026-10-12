# Builds and runs the C++ experiments of Chapter 30 (exercises 1, 2, 3, 5, 6, 7) with the MSYS2 CLANG64 toolchain.
#
#   powershell -ExecutionPolicy Bypass -File DOCS\PDFs\Guide\scripts\solutions\cpp\build_and_run.ps1
#
# Needs clang++ on PATH (C:\msys64\clang64\bin, which also provides the ASan runtime DLL) and the Eigen 3.4.0 sources that the
# project's release preset fetches into build\release\_deps\eigen-src. Executables go to the out\ folder next to this script.
# Several experiments fail on purpose (sanitizer reports, Eigen's malloc assertion, compile errors): read the output, not the exit codes.

$ErrorActionPreference = "Continue"
$here  = $PSScriptRoot
$repo  = (Resolve-Path (Join-Path $here "..\..\..\..\..\..")).Path
$eigen = Join-Path $repo "build\release\_deps\eigen-src"
$tests = Join-Path $repo "tests"                       # for support/alloc_guard.hpp
$out   = Join-Path $here "out"
if (-not (Test-Path $eigen)) { Write-Error "Eigen not found at $eigen; configure the release preset first (cmake --preset release)"; exit 1 }
New-Item -ItemType Directory -Force $out | Out-Null

$san = @("-g", "-O0", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-sanitize-recover=all", "-DEIGEN_RUNTIME_NO_MALLOC")
$rel = @("-O3", "-DNDEBUG")
$env:UBSAN_OPTIONS = "print_stacktrace=1:halt_on_error=1"

function Build($name, $flags, $suffix) {
    & clang++ -std=c++20 -isystem $eigen -I $tests @flags (Join-Path $here "$name.cpp") -o (Join-Path $out "$name`_$suffix.exe") 2>&1 | Select-Object -First 5
}
function Run($exe, $argList) {
    Write-Output "--- $exe $argList"
    $o = cmd /c "`"$(Join-Path $out $exe)`" $argList 2>&1"
    $o | Select-Object -First 6
    Write-Output "    (exit code $LASTEXITCODE)"
}

foreach ($n in @("ex1_alloc_guard", "ex2_auto", "ex3_timing", "ex5_undefined_behavior", "ex6_ref_rowmajor", "ex6_ref_colmajor")) {
    Build $n $rel "rel"
    Build $n $san "san"
}
Build "ex5_undefined_behavior" @("-O3", "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST") "hard"

Write-Output "=== 30.1 allocation guard"
Run "ex1_alloc_guard_rel.exe" ""
Run "ex1_alloc_guard_san.exe" ""
Write-Output "=== 30.2 auto with Eigen expressions (cases 1-5)"
foreach ($k in 1..5) { Run "ex2_auto_rel.exe" "$k"; Run "ex2_auto_san.exe" "$k" }
Write-Output "=== 30.3 covariance propagation timing"
Run "ex3_timing_rel.exe" "10000"
Run "ex3_timing_san.exe" "10000"
Write-Output "=== 30.5 out-of-bounds read and signed overflow"
Run "ex5_undefined_behavior_rel.exe" ""
Run "ex5_undefined_behavior_rel.exe" "int"
Run "ex5_undefined_behavior_san.exe" ""
Run "ex5_undefined_behavior_san.exe" "int"
Run "ex5_undefined_behavior_hard.exe" ""
Write-Output "=== 30.6 Eigen::Ref with a row-major and a column-major block"
Run "ex6_ref_rowmajor_san.exe" ""
Run "ex6_ref_colmajor_san.exe" ""
Write-Output "--- writable Ref (expected compile error)"
& clang++ -std=c++20 -isystem $eigen -fsyntax-only (Join-Path $here "ex6_ref_writable_compile_error.cpp") 2>&1 | Select-String "error" | Select-Object -First 3
Write-Output "=== 30.7 non-exhaustive std::visit (expected compile error)"
& clang++ -std=c++20 -fsyntax-only (Join-Path $here "ex7_variant_compile_error.cpp") 2>&1 | Select-String "error" | Select-Object -First 3

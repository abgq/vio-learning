# ==============================================================================
# External Dependencies Configuration
# Downloads and configures third-party libraries via FetchContent.
# ==============================================================================
include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SYSTEM
)

set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(BUILD_TESTING OFF) 

FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
    SYSTEM
)

FetchContent_MakeAvailable(catch2)
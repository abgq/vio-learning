# ==============================================================================
# Compiler Warnings Configuration
# Enforces strict code quality checks for VIO internal targets.
# ==============================================================================
add_library(vio_warnings INTERFACE)
target_compile_options(vio_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion 
    -Wold-style-cast -Wnon-virtual-dtor -Wdouble-promotion -Werror -Wno-c2y-extensions
)

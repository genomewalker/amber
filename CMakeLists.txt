cmake_minimum_required(VERSION 3.18)
project(amber VERSION 1.0.0 LANGUAGES C CXX)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Git-based version detection
# ---------------------------------------------------------------------------
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_RESULT
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_RESULT EQUAL 0 AND GIT_VERSION)
        string(REGEX REPLACE "^v" "" AMBER_VERSION "${GIT_VERSION}")
    else()
        set(AMBER_VERSION "${PROJECT_VERSION}-${GIT_COMMIT}")
    endif()
else()
    set(AMBER_VERSION "${PROJECT_VERSION}")
    set(GIT_COMMIT "unknown")
endif()

message(STATUS "AMBER version: ${AMBER_VERSION}")

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_BINARY_DIR}/include/amber/version.h
    @ONLY
)
include_directories(${CMAKE_BINARY_DIR}/include)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(AMBER_USE_TORCH "Build amber bin with LibTorch (CPU or GPU)" OFF)
option(AMBER_USE_LIBLEIDENALG "Build with libleidenalg for exact Leiden" ON)
option(AMBER_NATIVE_ARCH "Optimize for build machine CPU (-march=native)" OFF)

if(AMBER_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -funroll-loops")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")
endif()

# ---------------------------------------------------------------------------
# LibTorch (optional — enables amber bin; works with CPU or GPU LibTorch)
# ---------------------------------------------------------------------------
if(AMBER_USE_TORCH)
    # Allow newer GCC with CUDA (CUDA 11.7 + GCC 12)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_definitions(-DUSE_LIBTORCH)
    message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
else()
    message(STATUS "AMBER_USE_TORCH=OFF — amber bin will print a build error at runtime")
endif()

# ---------------------------------------------------------------------------
# libleidenalg (optional but strongly recommended)
# ---------------------------------------------------------------------------
if(AMBER_USE_LIBLEIDENALG)
    message(STATUS "Fetching igraph 0.10.17 from GitHub...")
    FetchContent_Declare(
        igraph
        GIT_REPOSITORY https://github.com/igraph/igraph.git
        GIT_TAG        0.10.17
        GIT_SHALLOW    TRUE
        OVERRIDE_FIND_PACKAGE
    )
    set(IGRAPH_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(IGRAPH_ENABLE_TLS ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_BLAS ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_LAPACK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_ARPACK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_GLPK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_GMP ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_PLFIT ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(igraph)

    if(NOT TARGET igraph::igraph)
        add_library(igraph::igraph ALIAS igraph)
    endif()

    message(STATUS "Fetching libleidenalg v0.9.0 from GitHub...")
    FetchContent_Declare(
        libleidenalg
        GIT_REPOSITORY https://github.com/vtraag/libleidenalg.git
        GIT_TAG        v0.9.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(libleidenalg)
    if(NOT libleidenalg_POPULATED)
        FetchContent_Populate(libleidenalg)

        set(LIBLEIDENALG_SOURCES
            ${libleidenalg_SOURCE_DIR}/src/GraphHelper.cpp
            ${libleidenalg_SOURCE_DIR}/src/Optimiser.cpp
            ${libleidenalg_SOURCE_DIR}/src/MutableVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/ModularityVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/SignificanceVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/SurpriseVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/ResolutionParameterVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/LinearResolutionParameterVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/CPMVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/RBConfigurationVertexPartition.cpp
            ${libleidenalg_SOURCE_DIR}/src/RBERVertexPartition.cpp
        )

        add_library(libleidenalg STATIC ${LIBLEIDENALG_SOURCES})

        include(GenerateExportHeader)
        generate_export_header(libleidenalg
            STATIC_DEFINE LEIDENALG_STATIC
            EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/libleidenalg_export.h
        )

        target_include_directories(libleidenalg BEFORE PUBLIC
            ${CMAKE_CURRENT_BINARY_DIR}
            ${libleidenalg_SOURCE_DIR}/include
            ${libleidenalg_SOURCE_DIR}/include/libleidenalg
        )
        target_compile_definitions(libleidenalg PUBLIC LEIDENALG_STATIC)
        target_link_libraries(libleidenalg PUBLIC igraph)
        set_target_properties(libleidenalg PROPERTIES CXX_STANDARD 11)
    endif()

    if(NOT TARGET libleidenalg::libleidenalg)
        add_library(libleidenalg::libleidenalg ALIAS libleidenalg)
    endif()

    set(LIBLEIDENALG_INCLUDE_DIR "${libleidenalg_SOURCE_DIR}/include" CACHE PATH "" FORCE)
    set(IGRAPH_INCLUDE_DIR "${igraph_SOURCE_DIR}/include" CACHE PATH "" FORCE)
    set(IGRAPH_BINARY_INCLUDE_DIR "${igraph_BINARY_DIR}/include" CACHE PATH "" FORCE)

    add_definitions(-DUSE_LIBLEIDENALG)
    message(STATUS "Using igraph + libleidenalg from GitHub")
endif()

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(HTSLIB htslib)
endif()

if(NOT HTSLIB_FOUND)
    if(DEFINED ENV{CONDA_PREFIX})
        set(HTSLIB_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
        set(HTSLIB_LIBRARY_DIRS "$ENV{CONDA_PREFIX}/lib")
        set(HTSLIB_LIBRARIES "-lhts")
    else()
        message(FATAL_ERROR "HTSlib not found. Install via conda or set PKG_CONFIG_PATH.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Include / link dirs
# ---------------------------------------------------------------------------
if(AMBER_USE_LIBLEIDENALG AND LIBLEIDENALG_INCLUDE_DIR)
    include_directories(BEFORE
        ${LIBLEIDENALG_INCLUDE_DIR}
        ${LIBLEIDENALG_INCLUDE_DIR}/libleidenalg
        ${IGRAPH_INCLUDE_DIR}
        ${IGRAPH_BINARY_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${HTSLIB_INCLUDE_DIRS}
)

if(AMBER_USE_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

link_directories(${HTSLIB_LIBRARY_DIRS})

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(AMBER_CORE_SOURCES
    src/main.cpp
    src/cli/cli_common.cpp
    src/cli/cmd_deconvolve.cpp
    src/cli/cmd_chimera.cpp
    src/cli/cmd_seeds.cpp
    src/cli/cmd_resolve.cpp
    src/cli/cmd_damage.cpp
    src/seeds/seed_generator.cpp
    src/polish.cpp
    src/chimera.cpp
    src/algorithms/fractal_features.cpp
    src/algorithms/kmer_features.cpp
    src/algorithms/coverage_features.cpp
    src/algorithms/damage_features.cpp
    src/algorithms/damage_profile.cpp
    src/algorithms/sequence_utils.cpp
    src/algorithms/pca.cpp
    src/algorithms/chimera_detector.cpp
    src/algorithms/ref_reader.cpp
    src/algorithms/parallel_coverage.cpp
    src/binner/binner.cpp
    src/bin2/clustering/hnsw_knn_index.cpp
    src/bin2/clustering/edge_weighter.cpp
    src/bin2/clustering/leiden_backend.cpp
    src/bin2/clustering/marker_index.cpp
    src/bin2/clustering/quality_leiden_backend.cpp
    src/bin2/clustering/consensus_knn.cpp
    src/bin2/quality/marker_quality.cpp
    src/bin2/quality/checkm_markers.cpp
)

if(AMBER_USE_TORCH)
    # Full bin implementation (neural network training, GPU or CPU LibTorch)
    list(APPEND AMBER_CORE_SOURCES
        src/bin2.cpp
        src/cli/cmd_bin.cpp
    )
else()
    # Stubs: amber bin and amber damage print informative errors and exit
    list(APPEND AMBER_CORE_SOURCES
        src/cli/cmd_bin_stub.cpp
        src/algorithms/damage_extraction_stub.cpp
    )
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(amber ${AMBER_CORE_SOURCES})

target_link_libraries(amber
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${HTSLIB_LIBRARIES}
    z
    pthread
    m
)

if(AMBER_USE_LIBLEIDENALG)
    target_link_libraries(amber
        igraph::igraph
        libleidenalg::libleidenalg
    )
endif()

if(AMBER_USE_TORCH)
    target_link_libraries(amber ${TORCH_LIBRARIES})
    if(TORCH_CUDA_LIBRARIES)
        target_link_libraries(amber ${TORCH_CUDA_LIBRARIES})
    endif()
endif()

set_property(TARGET amber PROPERTY CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS amber DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests (BUILD_TESTING is defined by include(CTest) above)
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
    message(STATUS "Tests enabled")
    # Smoke test: binary must exit cleanly with --version
    add_test(NAME smoke_version COMMAND amber --version)
    add_test(NAME smoke_help    COMMAND amber --help)
    set_tests_properties(smoke_help PROPERTIES WILL_FAIL FALSE PASS_REGULAR_EXPRESSION "amber")
endif()

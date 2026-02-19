cmake_minimum_required(VERSION 3.18)
project(amber_gpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Option to enable LibTorch
option(AMBER_USE_TORCH "Build with LibTorch support" ON)

# Option to use original libleidenalg (exact COMEBin matching)
option(AMBER_USE_LIBLEIDENALG "Build with libleidenalg for exact Leiden" ON)

if(AMBER_USE_TORCH)
    # Allow newer GCC with CUDA (CUDA 11.7 + GCC 12)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    add_definitions(-DUSE_LIBTORCH)
endif()

if(AMBER_USE_LIBLEIDENALG)
    message(STATUS "Fetching igraph 0.10.17 from GitHub...")
    FetchContent_Declare(
        igraph
        GIT_REPOSITORY https://github.com/igraph/igraph.git
        GIT_TAG        0.10.17
        GIT_SHALLOW    TRUE
        OVERRIDE_FIND_PACKAGE
    )
    # igraph build options - minimal build
    set(IGRAPH_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(IGRAPH_ENABLE_TLS OFF CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_BLAS ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_LAPACK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_ARPACK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_GLPK ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_GMP ON CACHE BOOL "" FORCE)
    set(IGRAPH_USE_INTERNAL_PLFIT ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(igraph)

    # Create alias target that libleidenalg expects
    if(NOT TARGET igraph::igraph)
        add_library(igraph::igraph ALIAS igraph)
    endif()

    # Fetch libleidenalg but build manually to avoid export issues
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

        # Build libleidenalg manually (avoiding its install/export logic)
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

        # Generate export header (normally done by libleidenalg's CMakeLists.txt)
        include(GenerateExportHeader)
        generate_export_header(libleidenalg
            STATIC_DEFINE LEIDENALG_STATIC
            EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/libleidenalg_export.h
        )

        # Use BEFORE to ensure FetchContent headers take precedence over conda
        target_include_directories(libleidenalg BEFORE PUBLIC
            ${CMAKE_CURRENT_BINARY_DIR}  # For libleidenalg_export.h
            ${libleidenalg_SOURCE_DIR}/include
            ${libleidenalg_SOURCE_DIR}/include/libleidenalg
        )
        target_compile_definitions(libleidenalg PUBLIC LEIDENALG_STATIC)
        target_link_libraries(libleidenalg PUBLIC igraph)
        set_target_properties(libleidenalg PROPERTIES CXX_STANDARD 11)
    endif()

    # Create alias for libleidenalg
    if(NOT TARGET libleidenalg::libleidenalg)
        add_library(libleidenalg::libleidenalg ALIAS libleidenalg)
    endif()

    # Store paths in CACHE for use later
    set(LIBLEIDENALG_INCLUDE_DIR "${libleidenalg_SOURCE_DIR}/include" CACHE PATH "" FORCE)
    set(IGRAPH_INCLUDE_DIR "${igraph_SOURCE_DIR}/include" CACHE PATH "" FORCE)
    set(IGRAPH_BINARY_INCLUDE_DIR "${igraph_BINARY_DIR}/include" CACHE PATH "" FORCE)

    add_definitions(-DUSE_LIBLEIDENALG)
    message(STATUS "Using igraph + libleidenalg from GitHub")
    message(STATUS "  libleidenalg headers: ${libleidenalg_SOURCE_DIR}/include")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find HTSlib via pkg-config or environment
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(HTSLIB htslib)
endif()

if(NOT HTSLIB_FOUND)
    # Fall back to conda env
    if(DEFINED ENV{CONDA_PREFIX})
        set(HTSLIB_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
        set(HTSLIB_LIBRARY_DIRS "$ENV{CONDA_PREFIX}/lib")
        set(HTSLIB_LIBRARIES "-lhts")
    else()
        message(FATAL_ERROR "HTSlib not found")
    endif()
endif()

# FetchContent directories must come BEFORE conda to avoid header conflicts
# Use non-SYSTEM includes (-I) which take precedence over -isystem from conda compiler
if(AMBER_USE_LIBLEIDENALG AND LIBLEIDENALG_INCLUDE_DIR)
    include_directories(BEFORE
        ${LIBLEIDENALG_INCLUDE_DIR}
        ${LIBLEIDENALG_INCLUDE_DIR}/libleidenalg
        ${IGRAPH_INCLUDE_DIR}
        ${IGRAPH_BINARY_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${HTSLIB_INCLUDE_DIRS}
)

if(AMBER_USE_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

link_directories(${HTSLIB_LIBRARY_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/cli/cli_common.cpp
    src/cli/cmd_bin.cpp
    src/cli/cmd_bin2.cpp
    src/cli/cmd_deconvolve.cpp
    src/cli/cmd_polish.cpp
    src/cli/cmd_chimera.cpp
    src/cli/cmd_seeds.cpp
    src/seeds/seed_generator.cpp
    src/polish.cpp
    src/chimera.cpp
    src/bin2.cpp
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
    src/bin2/clustering/ensemble_leiden.cpp
    src/bin2/clustering/marker_index.cpp
    src/bin2/clustering/quality_leiden_backend.cpp
    src/bin2/clustering/consensus_knn.cpp
    src/bin2/quality/marker_quality.cpp
    src/bin2/quality/checkm_markers.cpp
)

# Build executable
add_executable(amber_gpu ${SOURCES})

target_link_libraries(amber_gpu
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${HTSLIB_LIBRARIES}
    z
    pthread
    m
)

if(AMBER_USE_LIBLEIDENALG)
    # Link to FetchContent-built libraries
    target_link_libraries(amber_gpu
        igraph::igraph
        libleidenalg::libleidenalg
    )
endif()

if(AMBER_USE_TORCH)
    target_link_libraries(amber_gpu ${TORCH_LIBRARIES})
    if(TORCH_CUDA_LIBRARIES)
        target_link_libraries(amber_gpu ${TORCH_CUDA_LIBRARIES})
    endif()
endif()

set_property(TARGET amber_gpu PROPERTY CXX_STANDARD 17)

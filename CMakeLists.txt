cmake_minimum_required (VERSION 3.18)
project(LibMolGrid LANGUAGES C CXX)

#version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 5)
set (VERSION_PATCH 3)

#set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

option(BUILD_SHARED "Build shared library" ON)
option(BUILD_STATIC "Build static library" ON)
option(BUILD_COVERAGE "Build with code coverage" OFF)
option(LIBMOLGRID_ENABLE_CUDA "Build the upstream CUDA backend when available" ON)
option(LIBMOLGRID_ENABLE_METAL "Build the Apple Metal backend when available" ${APPLE})

if(${BUILD_CONTAINED_PYTHON})
  set(BUILD_STATIC ON)
  set(Boost_USE_STATIC_LIBS ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules;${PROJECT_SOURCE_DIR}/cmake")
include(cmake/git_revision.cmake)
include(cmake/safeguards.cmake)
include(GNUInstallDirs)
include(CheckLanguage)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

#dependencies
set(LIBMOLGRID_USE_CUDA OFF)
set(LIBMOLGRID_USE_METAL OFF)

if(LIBMOLGRID_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
      set(LIBMOLGRID_USE_CUDA ON)
    elseif(NOT APPLE OR NOT LIBMOLGRID_ENABLE_METAL)
      message(FATAL_ERROR "CUDA was enabled, but the CUDA toolkit was not found.")
    else()
      message(STATUS "CUDA toolkit not found. Falling back to the Metal backend.")
    endif()
  elseif(NOT APPLE OR NOT LIBMOLGRID_ENABLE_METAL)
    message(FATAL_ERROR "CUDA was enabled, but no CUDA compiler was found.")
  else()
    message(STATUS "CUDA compiler not found. Falling back to the Metal backend.")
  endif()
endif()

if(NOT LIBMOLGRID_USE_CUDA AND LIBMOLGRID_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "The Metal backend is only available on Apple platforms.")
  endif()
  enable_language(OBJCXX)
  set(LIBMOLGRID_USE_METAL ON)
endif()

if(NOT LIBMOLGRID_USE_CUDA AND NOT LIBMOLGRID_USE_METAL)
  message(FATAL_ERROR "No supported GPU backend is enabled. Enable CUDA or Metal.")
endif()

find_package(Boost REQUIRED COMPONENTS regex unit_test_framework program_options filesystem iostreams serialization)
find_package(OpenBabel3 REQUIRED)
include_directories(SYSTEM ${OPENBABEL3_INCLUDE_DIR})
find_package(ZLIB)

if(LIBMOLGRID_USE_CUDA)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

if(LIBMOLGRID_USE_METAL)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

# configure a header file to pass some of the CMake settings
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/include/libmolgrid/config.h"
  )

set(CMAKE_CXX_STANDARD 17)
if(LIBMOLGRID_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES "all-major")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -g -lineinfo")
  set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")
endif()
if(LIBMOLGRID_USE_METAL)
  set(CMAKE_OBJCXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # OpenBabel 3.1.1 uses std::binary_function which was removed in C++17.
    # Restore it via the compatibility shim in libc++.
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
    if (BUILD_COVERAGE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fprofile-arcs -ftest-coverage")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    endif()
endif()

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(docs)

# enable testing
include(CTest)
enable_testing()
add_subdirectory(test)

cmake_minimum_required (VERSION 3.20)
project(llarr)
if (WIN32)
  set(OS Windows)
elseif(UNIX AND NOT APPLE)
  set(OS Linux)
elseif(APPLE)
  set(OS macOS)
endif()
include(FetchContent)
set(FETCHCONTENT_FULLY_DISCONNECTED OFF)
if (TRUE)
  FetchContent_Populate(
    vcpkg
    GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/vcpkg
    EXCLUDE_FROM_ALL TRUE
  )
endif()
if (EXISTS ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake)
  if (OS STREQUAL "Windows")
    set(VCPKG_TARGET_TRIPLET x64-windows-static-md CACHE STRING "")
  elseif(OS STREQUAL "Linux")
    set(VCPKG_TARGET_TRIPLET x64-linux CACHE STRING "")
  elseif(OS STREQUAL "macOS")
    set(VCPKG_TARGET_TRIPLET x64-osx CACHE STRING "")
  endif()
  include(${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake)
endif()

set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/external CACHE STRING "")
set(CPM_DOWNLOAD_VERSION 0.34.3) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_SOURCE_DIR}/external/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
  file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

FetchContent_Populate(
  ccache
  URL https://github.com/cristianadam/ccache/releases/download/v4.5/${OS}.tar.xz
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/ccache
  EXCLUDE_FROM_ALL TRUE
)
find_program(CCACHE_PROGRAM ccache PATHS ${ccache_SOURCE_DIR})
if (CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}" CACHE INTERNAL "")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE INTERNAL "")
endif()


CPMAddPackage(
  NAME gtest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.10.0
  VERSION 1.10.0
  OPTIONS "gtest_force_shared_crt ON" "BUILD_GMOCK OFF" "INSTALL_GTEST OFF"
)
CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 8.1.1
)
find_package(fmt)
add_subdirectory(src)
enable_testing()
add_subdirectory(tests)

cmake_minimum_required (VERSION 3.19)
project(llarr)

# bootstrap the package manager
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/external CACHE STRING "")
set(CPM_DOWNLOAD_VERSION 0.34.3) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_SOURCE_DIR}/external/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
  file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# setup os names
if (WIN32)
  set(OS Windows)
elseif(UNIX AND NOT APPLE)
  set(OS Linux)
elseif(APPLE)
  set(OS macOS)
endif()

# vcpkg
CPMAddPackage(
  NAME vcpkg
  GITHUB_REPOSITORY microsoft/vcpkg
  GIT_TAG 2022.02.02
  DOWNLOAD_ONLY TRUE
)
if (EXISTS ${CPM_PACKAGE_vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake)
  if (OS STREQUAL "Windows")
    set(VCPKG_TARGET_TRIPLET x64-windows-static-md CACHE STRING "")
  elseif(OS STREQUAL "Linux")
    set(VCPKG_TARGET_TRIPLET x64-linux CACHE STRING "")
  elseif(OS STREQUAL "macOS")
    set(VCPKG_TARGET_TRIPLET x64-osx CACHE STRING "")
  endif()
  include(${CPM_PACKAGE_vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake)
endif()

# ccache
CPMAddPackage(
  NAME ccache
  URL https://github.com/cristianadam/ccache/releases/download/v4.5/${OS}.tar.xz
  DOWNLOAD_ONLY TRUE
)
find_program(CCACHE_PROGRAM ccache PATHS ${CPM_PACKAGE_ccache_SOURCE_DIR})
if (CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}" CACHE INTERNAL "")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE INTERNAL "")
endif()


CPMAddPackage(
  NAME gtest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.11.0
  OPTIONS "gtest_force_shared_crt ON" "BUILD_GMOCK OFF" "INSTALL_GTEST OFF"
)
CPMAddPackage("gh:fmtlib/fmt#8.1.1")
CPMAddPackage(
  NAME glfw
  GITHUB_REPOSITORY glfw/glfw
  GIT_TAG 3.3.6
  OPTIONS "GLFW_BUILD_DOCS OFF" "GLFW_INSTALL OFF"
)
CPMAddPackage("gh:KhronosGroup/Vulkan-Headers@1.3.204")

# imgui start
CPMAddPackage(
  NAME imgui
  GITHUB_REPOSITORY ocornut/imgui
  GIT_TAG docking
  DOWNLOAD_ONLY TRUE
)
add_library(imgui)
target_sources(imgui
  PUBLIC
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui.h
  PRIVATE
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imconfig.h
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui_internal.h
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imstb_rectpack.h
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imstb_textedit.h
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imstb_truetype.h
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui.cpp
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui_demo.cpp
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui_draw.cpp
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui_tables.cpp
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC
  $<BUILD_INTERFACE:${CPM_PACKAGE_imgui_SOURCE_DIR}>
)
add_library(imgui_glfw)
target_sources(imgui_glfw
  PUBLIC
     ${CPM_PACKAGE_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
  PRIVATE
     ${CPM_PACKAGE_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)
target_include_directories(imgui_glfw PUBLIC
  $<BUILD_INTERFACE:${CPM_PACKAGE_imgui_SOURCE_DIR}/backends>
)
target_link_libraries(imgui_glfw PUBLIC imgui glfw)
add_library(imgui_vulkan)
target_sources(imgui_vulkan
  PUBLIC
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.h
  PRIVATE
    ${CPM_PACKAGE_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_vulkan PUBLIC
  $<BUILD_INTERFACE:${CPM_PACKAGE_imgui_SOURCE_DIR}/backends>
)
target_compile_definitions(imgui_vulkan
  PUBLIC
    IMGUI_IMPL_VULKAN_NO_PROTOTYPES
    GLFW_INCLUDE_VULKAN
    GLFW_INCLUDE_NONE
)
target_link_libraries(imgui_vulkan PUBLIC imgui Vulkan::Headers)
# imgui end

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)

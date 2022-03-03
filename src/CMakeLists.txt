add_executable(main)
target_sources(main
  PRIVATE
    main.cpp
    vulkan.cpp
)
target_link_libraries(main PRIVATE fmt imgui_glfw imgui_vulkan)
target_compile_definitions(main
  PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC VULKAN_HPP_NO_CONSTRUCTORS
    VULKAN_HPP_NO_EXCEPTIONS
)
if (WIN32)
set_property(TARGET main PROPERTY CXX_STANDARD 23)
else()
set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()

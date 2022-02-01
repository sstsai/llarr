add_executable(main)
target_sources(main
  PRIVATE
    main.cpp
)
target_link_libraries(main PRIVATE fmt::fmt)
if (WIN32)
set_property(TARGET main PROPERTY CXX_STANDARD 23)
else()
set_property(TARGET main PROPERTY CXX_STANDARD 20)
endif()

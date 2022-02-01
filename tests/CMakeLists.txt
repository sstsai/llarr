add_executable(tests)
target_sources(tests
  PRIVATE
    main.cpp
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(tests PRIVATE fmt::fmt GTest::gtest_main)
if (WIN32)
set_property(TARGET tests PROPERTY CXX_STANDARD 23)
else()
set_property(TARGET tests PROPERTY CXX_STANDARD 20)
endif()

include(GoogleTest)
gtest_discover_tests(tests)

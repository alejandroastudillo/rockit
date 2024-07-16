cmake_minimum_required(VERSION 3.0.0)
project(interface_usage VERSION 0.1.0)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")

add_subdirectory(fatrop)

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")
set(CMAKE_BUILD_TYPE Release)
add_library(fatrop_driver SHARED fatrop_driver.cpp ../casadi_codegen.c)
set_property(SOURCE fatrop_driver.cpp APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/after_init.c.in)

target_link_libraries(fatrop_driver fatrop)
target_include_directories(fatrop_driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fatrop/fatrop/)

add_executable(example example.cpp ../casadi_codegen.c)
target_link_libraries(example fatrop)
target_include_directories(example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fatrop/fatrop/)


INSTALL(TARGETS fatrop_driver
  LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIBRARY_INSTALL_DIR}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
  RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
  )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

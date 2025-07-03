cmake_minimum_required(VERSION 3.0.2)
cmake_policy(SET CMP0079 NEW) # for CMake 3.19 and above
project(xbox_chassis)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_FLAGS_RELEASE "-O3")
# add the cmake submodule for convenience
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(message_type)
include(add_dependency)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  moveit_ctrl  # 依赖moveit_ctrl包来使用JoystickState消息
  message_runtime  # 添加这个以支持消息运行时
)

## System dependencies
find_package(Threads REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(Eigen3 REQUIRED)
###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs moveit_ctrl message_runtime
)

add_subdirectory(external)
add_subdirectory(main)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Specify libraries to link a library or executable target against
target_link_libraries(run PUBLIC
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)



#############
## Install ##
#############

## Mark executables for installation
install(TARGETS run
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

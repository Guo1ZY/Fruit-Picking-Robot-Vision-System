cmake_minimum_required(VERSION 3.0.2)
cmake_policy(SET CMP0079 NEW) # for CMake 3.19 and above
project(apple_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_FLAGS_RELEASE "-O3")
# add the cmake submodule for convenience
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(message_type)
include(add_dependency)

#指定cv_bridge_1DIR
# set(cv_bridge_DIR /home/zy/catkin_ws/devel/share/cv_bridge/cmake)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge_1
  image_transport
  message_filters
  tf2
  tf2_ros
  geometry_msgs
)

## Find OpenCV
find_package(OpenCV 4.7 REQUIRED)
find_package(Eigen3 REQUIRED)

## catkin specific configuration
catkin_package(
  INCLUDE_DIRS 
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    roscpp 
    rospy 
    std_msgs 
    sensor_msgs 
    cv_bridge_1
    image_transport 
    message_filters
    tf2
    tf2_ros
    geometry_msgs
  DEPENDS OpenCV
)

add_subdirectory(external)
add_subdirectory(main)
add_subdirectory(yolo)
## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)


## Specify libraries to link a library or executable target against
target_link_libraries(run PUBLIC
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

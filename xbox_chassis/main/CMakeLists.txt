file(GLOB_RECURSE main_srcs CONFIGURE_DEPENDS include/*.hpp include/*.h sources/*.cpp sources/*.c)

# add_library(drone STATIC ${main_srcs})
add_executable(run ${main_srcs} 
)

target_include_directories(run PUBLIC include)
target_include_directories(run PUBLIC ${catkin_INCLUDE_DIRS})

target_link_libraries(run PUBLIC external)
target_link_libraries(run PUBLIC ${catkin_LIBRARIES})
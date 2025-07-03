add_library(external external.h external.cpp)

add_subdirectory(serial) 

target_link_libraries(external PUBLIC serial)

cmake_minimum_required(VERSION 2.8)
project(LCEBDFTB)

set(CMAKE_BUILD_TYPE EQUAL "DEBUG")
find_package(OpenCV 3.2 REQUIRED)

find_package(MathGL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(barcodelocalizer main.cpp)
target_link_libraries(barcodelocalizer ${OpenCV_LIBS} ${MathGL_LIBRARIES})


cmake_minimum_required(VERSION 3.0.0)

set(OpenCV_DIR /home/linaro/workspace/opencv/lib/cmake/opencv4)  # 填入OpenCVConfig.cmake
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/bytetrack/include"
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/include
)

set(
    dynamic_libs
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/lib/librga.so
)

include_directories(${include_libs})
aux_source_directory(${PROJECT_SOURCE_DIR}/bytetrack/src BYTETRACK_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src BYTETRACK_SRC_DIR)

add_library(bytetrack ${BYTETRACK_SRC_DIR})

target_link_libraries(bytetrack ${OpenCV_LIBS})
target_link_libraries(bytetrack ${dynamic_libs})
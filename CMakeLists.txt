cmake_minimum_required(VERSION 3.0.0)
project(test VERSION 0.1.0)

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    /home/linaro/workspace/opencv/include/opencv4
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/include
)

include_directories(${include_libs})

add_executable(test yolov5_c.cpp videoio.cpp detect.cpp decode.cpp nms.cpp)

# 添加动态链接库opencv及pthread
set(opencv_lib "/home/linaro/workspace/opencv/lib")
set(
    dynamic_libs  pthread 
    ${opencv_lib}/libopencv_core.so
    ${opencv_lib}/libopencv_highgui.so
    ${opencv_lib}/libopencv_imgcodecs.so
    ${opencv_lib}/libopencv_imgproc.so
    ${opencv_lib}/libopencv_videoio.so
    ${opencv_lib}/libopencv_video.so
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/aarch64/librknnrt.so
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/lib/librga.so
)

# 开启调试选项
add_definitions("-g")

target_link_libraries(test ${dynamic_libs})
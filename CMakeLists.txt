cmake_minimum_required(VERSION 3.0.0)
project(yolov5_deepsort VERSION 0.1.0)

add_subdirectory(deepsort)

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/yolov5/include"
    "${PROJECT_SOURCE_DIR}/deepsort/include"
    /home/linaro/workspace/opencv/include/opencv4
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/3rdparty/rga/include
)

include_directories(${include_libs})

aux_source_directory(${PROJECT_SOURCE_DIR}/yolov5/src YOLO_SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src YOLO_SRC_DIR)
add_executable(yolov5_deepsort yolov5_deepsort.cpp ${YOLO_SRC_DIR})

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

target_link_libraries(yolov5_deepsort ${dynamic_libs} deepsort)
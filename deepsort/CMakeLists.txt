cmake_minimum_required(VERSION 3.0.0)

set(
    include_libs  
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/deepsort/include"
    /home/linaro/workspace/opencv/include/opencv4
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/include
)
set(opencv_lib "/home/linaro/workspace/opencv/lib")
set(
    dynamic_libs
    ${opencv_lib}/libopencv_core.so
    ${opencv_lib}/libopencv_highgui.so
    ${opencv_lib}/libopencv_imgcodecs.so
    ${opencv_lib}/libopencv_imgproc.so
    ${opencv_lib}/libopencv_videoio.so
    ${opencv_lib}/libopencv_video.so
    ${PROJECT_SOURCE_DIR}/3rdparty/librknn_api/aarch64/librknnrt.so
)

include_directories(${include_libs})
aux_source_directory(${PROJECT_SOURCE_DIR}/deepsort/src DEEPSORT_SRC_DIR)
# aux_source_directory(${PROJECT_SOURCE_DIR}/src DEEPSORT_SRC_DIR)

add_library(deepsort ${DEEPSORT_SRC_DIR})

target_link_libraries(deepsort ${dynamic_libs})
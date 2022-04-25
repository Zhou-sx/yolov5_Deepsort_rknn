#include "main.h"

/* 
    decode.cpp中的四个数据解码函数
    decode.cpp中有详细说明
*/
cv::Rect_<float> get_yolo_box(float *x, float *biases, int n, int index, int i, int j, 
                            int lw, int lh, int netw, int neth, int stride);

void get_network_boxes(float* predictions, int netw, int neth, int GRID, int* masks, 
                       float* anchors, int &box_off, detection* dets);

int outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets);
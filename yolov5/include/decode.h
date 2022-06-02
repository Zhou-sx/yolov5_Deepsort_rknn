#include "common.h"

// output type: fp32
int post_process_fp(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,int h_offset, 
                    int w_offset, float resize_scale, float conf_threshold, float nms_threshold, std::vector<Object>& objects);

int readLines(const char *fileName, char *lines[], int max_line);
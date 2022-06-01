#include "common.h"

// output type: uint8
int post_process_u8(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                 int h_offset, int w_offset, float resize_scale, float conf_threshold, float nms_threshold, 
                 std::vector<uint8_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

// output type: fp32
int post_process_fp(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,
                 int h_offset, int w_offset, float resize_scale, float conf_threshold, float nms_threshold, 
                 detect_result_group_t *group);

int readLines(const char *fileName, char *lines[], int max_line);
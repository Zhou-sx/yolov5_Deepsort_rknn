#ifndef RKNN_FP
#define RKNN_FP

#include "rknn_api.h"

class rknn_fp{
public:
    /*
        NPU初始化
        model_path： 模型路径
        thread_id：  线程号
        device_id：  设备号
    */
    rknn_fp(const char *, int, rknn_core_mask, int, int);
    ~rknn_fp(void);
    void dump_tensor_attr(rknn_tensor_attr*);
    int inference(unsigned char *);
    float cal_NPU_performance(std::queue<float> &, float &, float);
public:
    int _cpu_id;
    int _n_input;
    int _n_output;
    //Inputs and Output sets
    rknn_context ctx;
    rknn_tensor_attr _input_attrs[1];
    rknn_tensor_attr _output_attrs[3];
    rknn_tensor_mem* _input_mems[1];
    rknn_tensor_mem* _output_mems[3];
    float* _output_buff[3];
};

#endif
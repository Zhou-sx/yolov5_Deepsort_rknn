#include "main.h"
#include "decode.h"
#include "nms.h"

extern bool bReading;
extern bool bDetecting;     // 目标检测进程状态
// 开始 结束时间 ns
extern double start_time;
extern double end_time;
extern video_property video_probs; // 视频属性类
// 多线程控制相关
extern int idxOutputImage;                // next frame index to be output 保证queueDetOut_server序号正确
extern mutex mtxQueueInput;               
extern queue<input_image> queueInput;  // input queue
extern mutex mtxqueueDetOut;
extern queue<det_res> queueDetOut;        // Det output queue


class rknn_fp{
public:
    //Inputs and Output sets
    rknn_context ctx;
    const int _n_input = 1;
    const int _n_output = 3;
    rknn_tensor_attr _input_attrs[1];
    rknn_tensor_attr _output_attrs[3];
    rknn_tensor_mem* _input_mems[1];
    rknn_tensor_mem* _output_mems[3];
    float* _output_buff[3];
    /*
        NPU初始化
        model_path： 模型路径
        thread_id：  线程号
        device_id：  设备号
    */
    void dump_tensor_attr(rknn_tensor_attr* attr);
    rknn_fp(const char *, int, rknn_core_mask);
    ~rknn_fp(void);
    int detect(cv::Mat);
private:
    
};

float cal_NPU_performance(queue<float> &history_time, float &sum_time, float current_time);
int detect_process(const char *model_path, int cpuid, rknn_core_mask); 
// int draw_image(cv::Mat img,float scale,detection* dets,int total,float thresh); //画图函数
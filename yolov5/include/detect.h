#include "main.h"
#include "decode.h"
#include "rknn_fp.h"

extern bool bReading;
extern bool bDetecting;     // 目标检测进程状态
// 开始 结束时间 ns
extern double start_time;
extern double end_time;
extern video_property video_probs; // 视频属性类
// 多线程控制相关
extern int idxOutputImage;            // next frame index to be output 保证queueDetOut_server序号正确
extern mutex mtxQueueInput;               
extern queue<input_image> queueInput;  // input queue
extern mutex mtxqueueDetOut;
extern queue<imageout_idx> queueDetOut;// Det output queue


class Yolo :public rknn_fp{
public:
    using rknn_fp::rknn_fp;  //声明使用基类的构造函数
    float cal_NPU_performance(queue<float> &history_time, float &sum_time, float current_time);
    int detect_process();
};


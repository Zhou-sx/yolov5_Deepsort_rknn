#include "main.h"
#include "decode.h"
#include "nms.h"

extern bool bReading;
extern const int NPU_threads; //NPU线程数量
extern int NPU_ready;   //就绪状态的NPU线程数量
extern int NPU_finish;   //就绪状态的NPU线程数量
// 多线程控制相关
extern int idxOutputImage;                // next frame index to be output 保证queueDetOut_server序号正确
extern mutex mtxQueueInput;               // mutex of input queue
extern queue<pair<int, Mat>> queueInput;  // input queue
extern mutex mtxqueueDetOut;
extern queue<det_res> queueDetOut; // output queue

/*
	图像预处理 
	将图片缩放至指定大小，不进行图像拉伸保持同一比例
	img——原图像      _img——缩放后的图像
	resize：  		缩放图像
	get_max_scale： 获取最大缩放比例
*/
class preprocess
{
public:
    preprocess();
    void set_size(int height, int width);//输入变化至的尺寸 height，width
    int input_height;
    int input_width;
	float scale;
	void resize(cv::Mat &img, cv::Mat &_img);

private:
	float get_max_scale(cv::Mat &img);
};

class rknn_fp{
public:
    //Inputs and Output sets
	rknn_input inputs[1];
	rknn_output outputs[3];
	rknn_tensor_attr outputs_attr[3];
    rknn_context ctx;
    // 预处理类
    preprocess post_do;
    /*
        NPU初始化
        model_path： 模型路径
        thread_id：  线程号
        device_id：  设备号
    */
    rknn_fp(const char *, int, rknn_core_mask);
    ~rknn_fp(void);
    int detect(cv::Mat);
private:
    
};

float cal_NPU_performance(queue<float> &history_time, float &sum_time, float current_time);
int detect_process(const char *model_path, int cpuid, rknn_core_mask); 
int draw_image(cv::Mat img,float scale,detection* dets,int total,float thresh); //画图函数
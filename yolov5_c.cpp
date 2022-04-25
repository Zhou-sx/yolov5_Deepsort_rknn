#include <stdio.h>
#include <thread>
#include "main.h"
#include "videoio.h"
#include "detect.h"

string PROJECT_DIR = "/home/linaro/workspace/yolov5_c";
string MODEL_PATH = PROJECT_DIR + "/model/yolov5_nofocus.rknn";
char* DEVICE_ID = (char*)"a056846056897056";
string IMAGE_PATH = PROJECT_DIR + "/data/DJI_0001_S_0.jpg";
string VIDEO_PATH = PROJECT_DIR + "/data/Track_res.mp4";
string VIDEO_SAVEPATH = PROJECT_DIR + "/data/results.mp4";

// 各任务进行状态序号
int idxInputImage = 0;  // image index of input video
int idxOutputImage = 0; // image index of output video
int idxShowImage = 0;	// 目标追踪下一帧要处理的对象
int Frame_cnt = 0;      // 视频总长度
bool bReading = true;   // flag of input
bool bDetecting = true; // 目标检测进程状态
double Time_video = 0;  // 整个视频追踪所花的时间
int NPU_ID[3] = {1, 2, 4};

// 多线程控制相关
mutex mtxQueueInput;        		  // mutex of input queue client
queue<pair<int, Mat>> queueInput;  // input queue client
mutex mtxqueueDetOut;
queue<det_res> queueDetOut; // output queue
mutex mtxQueueShow;                // mutex of display queue
priority_queue<imageout_idx, vector<imageout_idx>, paircomp> queueShow;  // display queue 目标追踪的输入
mutex mtxQueueOutput;			  // mutex of output queue client 最终输出
queue<Mat> queueOutput;  		   		  // output queue 目标追踪输出队列

double what_time_is_it_now()
{
	// 单位: ms
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * .001;
}

int main() {
  // cv::Mat img_src = cv::imread(IMAGE_PATH);
  // detect_process(MODEL_PATH.c_str(), 0, img_src);

  videoRead(VIDEO_PATH.c_str(), 7);
  const int thread_num = 5;
	array<thread, thread_num> threads;
	threads = {
                thread(videoRead, VIDEO_PATH.c_str(), 6),
                thread(videoWrite, VIDEO_SAVEPATH.c_str(), 7),
                thread(detect_process, MODEL_PATH.c_str(), 0, RKNN_NPU_CORE_0),
                thread(detect_process, MODEL_PATH.c_str(), 1, RKNN_NPU_CORE_1),
                thread(detect_process, MODEL_PATH.c_str(), 2, RKNN_NPU_CORE_2)
			  };
    for (int i = 0; i < thread_num; i++)
		threads[i].join();

    return 0;
}

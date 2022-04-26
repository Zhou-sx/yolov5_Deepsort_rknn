#include "main.h"
#include "opencv2/videoio/videoio_c.h"
using namespace std;

extern mutex mtxQueueInput;
extern queue<pair<int, Mat>> queueInput;  // input queue client
extern mutex mtxqueueDetOut;
extern queue<det_res> queueDetOut;        // Det output queue
extern mutex mtxQueueOutput;
extern queue<Mat> queueOutput;  // 目标追踪输出队列

extern bool bReading;      // flag of input
extern bool bDetecting;    // 目标检测进程状态
extern int idxInputImage;  // image index of input video

extern video_property video_probs; // 视频属性类

void videoRead(const char* video_name, int cpuid);
float get_max_scale(int input_width, int input_height, int net_width, int net_height);
void videoWrite(const char* save_path,int cpuid) ;
int draw_image(cv::Mat img,float scale,detection* dets,int total,float thresh);
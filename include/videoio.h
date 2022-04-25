#include "main.h"
#include "opencv2/videoio/videoio_c.h"
using namespace std;

extern mutex mtxQueueInput;        // mutex of input queue client
extern queue<pair<int, Mat>> queueInput;  // input queue client
extern mutex mtxQueueOutput;
extern queue<Mat> queueOutput;  // 目标追踪输出队列

extern bool bReading;      // flag of input
extern bool bDetecting;    // 目标检测进程状态
extern int Frame_cnt;
extern int idxInputImage;  // image index of input video

// extern video_property video_probs; // 视频属性类

void videoRead(const char* video_name, int cpuid);
void videoWrite(const char* save_path,int cpuid) ;
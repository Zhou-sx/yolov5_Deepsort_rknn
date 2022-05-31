#include "opencv2/videoio/videoio_c.h"

struct _detect_result_group_t;  // 前置声明

// 视频属性类
struct video_property{
    int Frame_cnt;
    int Fps;
    int Video_width;
    int Video_height;
    double Video_fourcc;
};


void videoRead(const char* video_name, int cpuid);
void videoResize(int cpuid);
void get_max_scale(int , int , int , int , double &, double &);
void videoWrite(const char* save_path,int cpuid) ;
int draw_image(cv::Mat& , struct _detect_result_group_t);

#include "opencv2/videoio/videoio_c.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"

struct _detect_result_group_t;  // 前置声明

// 视频属性类
struct video_property{
    int Frame_cnt;
    int Fps;
    int Video_width;
    int Video_height;
    double Video_fourcc;
};

/*
	图像预处理 
	img——原图像      _img——缩放后的图像
	resize：  		缩放图像
	get_max_scale： 获取最大缩放比例
*/
class preprocess
{
public:
    preprocess();
    int input_height;
    int input_width;
    int input_channel;
    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;

	void resize(cv::Mat &img, cv::Mat &_img);
};


void videoRead(const char* video_name, int cpuid);
void videoResize(int cpuid);
std::vector<float> get_max_scale(int input_width, int input_height, int net_width, int net_height);
void videoWrite(const char* save_path,int cpuid) ;
int draw_image(cv::Mat& , struct _detect_result_group_t);

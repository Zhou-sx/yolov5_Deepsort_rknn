#include "opencv2/opencv.hpp"
#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"

/*
	图像预处理 
	img——原图像      _img——缩放后的图像
	resize：  		缩放图像
	get_max_scale： 获取最大缩放比例
*/
class PreResize
{
public:
    PreResize();
    PreResize(int, int, int);
    void init(double, double);
    int input_height;
    int input_width;
    int input_channel;
    double fx;  // scale along x
    double fy;  // scale along y
    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;

	void resize(cv::Mat &img, cv::Mat &_img);
};


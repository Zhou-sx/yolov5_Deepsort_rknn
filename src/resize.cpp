#include "resize.h"

PreResize::PreResize(){
    
}

PreResize::PreResize(int height, int width, int channel){
	input_height = height;
	input_width = width;
	input_channel = channel;
}

void PreResize::init(double fx, double fy){
    this->fx = fx;
    this->fy = fy;
}

void PreResize::resize(cv::Mat &img, cv::Mat &_img)
{
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
	int img_width = img.cols;
    int img_height = img.rows;
    void *resize_buf = malloc(input_height * input_width * input_channel);

	src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, input_width, input_height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);

	IM_STATUS STATUS = imresize(src, dst);
    if (ret != IM_STATUS_NOERROR)
    {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
	_img = cv::Mat(cv::Size(input_width, input_height), CV_8UC3, resize_buf);
    free(resize_buf);
	// cv::imwrite("resize_input.jpg", _img);
}
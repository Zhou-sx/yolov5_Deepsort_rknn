/*
    客户端和服务器端的共享变量设置
    warning:不包含rknn_api相关设置
*/
#pragma once // 防止重定义 这个头文件会被多次引用
#include <unistd.h>
#include <mutex>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <set>
#include <queue>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
using namespace std;
using namespace cv;


#define BYTE unsigned char
#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080


// 网络的参数
#define NET_INPUTHEIGHT 640
#define NET_INPUTWIDTH 640
#define NET_INPUTCHANNEL 3
#define GRID0 80
#define GRID1 40
#define GRID2 20
#define nclasses 2
#define nyolo 3   //n yolo layers;
#define nanchor 3 //n anchor per yolo layer

// 阈值
#define OBJ_THRESH 0.3
#define NMS_THRESH 0.2
#define DRAW_CLASS_THRESH 0.3

// 每一层nbox的数量
#define nboxes_0 GRID0*GRID0*nanchor
#define nboxes_1 GRID1*GRID1*nanchor
#define nboxes_2 GRID2*GRID2*nanchor
#define nboxes_total nboxes_0+nboxes_1+nboxes_2

// 视频属性类
struct video_property{
    int Frame_cnt;
    int Fps;
    int Video_width;
    int Video_height;
    double Video_fourcc;
};

struct input_image{
    input_image(){

    }
    input_image(int num, cv::Mat img1, cv::Mat img2){
        index = num;
        img_src = img1;
        img_resize = img2;
    }
    int index;
    cv::Mat img_src;
    cv::Mat img_resize;
};

/*
    预测结果结构体:
    bbox:       存储box的位置信息 (x,y,w,h) 左上角坐标和宽、高
    prob:       预测结果置信度1 存在目标
    objectness: 预测结果置信度2 是某类目标的置信度
    sort_class: NMS中表示选哪一类进行排序;在其他情况下表示预测类别
    id:         目标ID号 用于追踪 辨别同一类及不同类的目标
*/
struct detection{
    cv::Rect_<float> bbox; 
    float prob[nclasses];
    float objectness;
    int sort_class;
	int id;    
};

/*
    某一帧的所有合理的预测结果
*/
struct det_res{
	int idx;
	int nboxes_left;
    cv::Mat img;
	detection res[nboxes_total];//缓冲区
};

/*
    某一帧的所有合理预测结果 + 图片
    frame_id：第x帧
    img：     背景图
    dets：    检测结果结构体数组
*/ 
struct imageout_idx
{
	int frame_id; 
	cv::Mat img; 
	vector<detection> dets;
};
/*
    输出队列
    按照帧数frame_id排序
*/ 
class paircomp {
public:
    bool operator()(const imageout_idx &n1, const imageout_idx &n2) const {
        return n1.frame_id > n2.frame_id;
    }
};


extern double what_time_is_it_now();
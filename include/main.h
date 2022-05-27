/*
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

#ifndef BOX_H
#include "box.h"
#define BOX_H
#endif // BOX_H

using namespace std;
using namespace cv;


#define BYTE unsigned char
#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define IMG_CHANNEL 3
#define IMG_PAD 1920

// 网络的参数
#define NET_INPUTHEIGHT 640
#define NET_INPUTWIDTH 640
#define NET_INPUTCHANNEL 3
#define GRID0 80
#define GRID1 40
#define GRID2 20
#define OBJ_CLASS_NUM     2
#define nyolo 3   //n yolo layers;
#define nanchor 3 //n anchor per yolo layer
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

// 阈值
#define NMS_THRESH        0.2
#define BOX_THRESH        0.5

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
        img_pad = img2;
    }
    int index;
    cv::Mat img_src;
    cv::Mat img_pad;
};


typedef struct _detect_result_group_t
{
    int id;
    int count;
    std::vector<DetectBox> results;
} detect_result_group_t;

/*
    某一帧的所有合理预测结果 + 图片
    img：     背景图
    dets：    检测结果结构体数组
*/ 
struct imageout_idx
{
	cv::Mat img; 
	detect_result_group_t dets;
};

// /*
//     输出队列
//     按照帧数frame_id排序
// */ 
// class paircomp {
// public:
//     bool operator()(const imageout_idx &n1, const imageout_idx &n2) const {
//         return n1.frame_id > n2.frame_id;
//     }
// };


extern double what_time_is_it_now();
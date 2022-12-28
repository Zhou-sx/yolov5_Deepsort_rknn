#pragma once // 防止重定义
#include <vector>
#include "opencv2/opencv.hpp"

#ifndef BOX_H
#include "box.h"
#define BOX_H
#endif // BOX_H

#define BYTE unsigned char
#define IMG_WIDTH 1024
#define IMG_HEIGHT 540
#define IMG_CHANNEL 3
#define IMG_PAD 640

// 网络的参数
#define NET_INPUTHEIGHT 640
#define NET_INPUTWIDTH 640
#define NET_INPUTCHANNEL 3
#define GRID0 80
#define GRID1 40
#define GRID2 20
#define OBJ_CLASS_NUM     1
#define nyolo 3   //n yolo layers;
#define nanchor 3 //n anchor per yolo layer
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

// 阈值
#define NMS_THRESH        0.2
#define BOX_THRESH        0.2

// 每一层nbox的数量
#define nboxes_0 GRID0*GRID0*nanchor
#define nboxes_1 GRID1*GRID1*nanchor
#define nboxes_2 GRID2*GRID2*nanchor
#define nboxes_total nboxes_0+nboxes_1+nboxes_2


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
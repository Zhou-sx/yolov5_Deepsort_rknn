#pragma once
#include "main.h"

// 计算IoU
double box_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
void swap(detection &a, detection &b);


/*
		NMS 针对于本问题优化了NMS方法
		即对每一类分别做nms 不会考虑行人和车辆之间的互相遮挡问题
*/
int nms_comparator(const void *pa, const void *pb);
int nms_sort(detection *dets, detection *nms_res, int total, int classes);
#include "nms.h"
/*
	计算IoU
*/
double box_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
/*
	用于Sort
*/
int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
/*
	NMS 针对于本问题优化了NMS方法
	即对每一类分别做nms 不会考虑行人和车辆之间的互相遮挡问题
    dets:    nms前
    nms_res: nms后的结果
    return:  nms剩下的大小
*/
int nms_sort(detection *dets, detection *nms_res, int total, int classes)
{
    int i, j, k;
    int cur = 0;
    for(k = 0; k < classes; ++k){
    	// for-loop 1: 不同的种类
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        // 置信度从高到低遍历
        for(i = 0; i < total; ++i){
            // 如果第i个置信度够则说明首先置信度本来就高且没有遮挡换到数组的最前面
            if(dets[i].prob[k] <= OBJ_THRESH){
				dets[i].prob[k] = 0;
				continue;
			}
            else{
                nms_res[cur++] = dets[i];
            }
            cv::Rect_<float> a = dets[i].bbox;
            // 0-i 不会拿来做iou
            for(j = i+1; j < total; ++j){
                cv::Rect_<float> b = dets[j].bbox;
				float iou = box_iou(a, b);
                if (iou > NMS_THRESH){
                    dets[j].prob[k] = 0;
					// dets[j].prob[k] *= (1.0 - iou); // 软NMS
                }
            }
        }
    }
	return cur;
}
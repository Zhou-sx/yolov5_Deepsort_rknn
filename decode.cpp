#include "decode.h"

/*---------------------------------------------------------
	Yolov5 输出结果解码1
	预测数据解码从0-1的相对数据变成像素值为单位的绝对数据
----------------------------------------------------------*/
cv::Rect_<float> get_yolo_box(float *x, float *biases, int n, int index, int i, int j,
                                int lw, int lh, int netw, int neth, int stride)
{
    cv::Rect_<float> b;
    //Yolov5
	b.x = (i - 0.5 + x[index + 0] * 2) / lw;
	b.y = (j - 0.5 + x[index + 1] * 2) / lh;
	float tmp = x[index + 2] * 2;
	b.width = tmp * tmp * biases[2 * n] / netw;
	tmp = x[index + 3] * 2;
	b.height = tmp * tmp * biases[2 * n + 1] / neth;
	b.x -= b.width / 2.0;
	b.y -= b.height / 2.0;
    return b;
}


/*---------------------------------------------------------
	Yolov5 输出结果解码2
	NPU推理结束会将所有的数据都存在一个数组里
	将乱序数组整理成一个个detection结构体 存放在结构体数组dets中
	同时会剔除置信度过低的预测结果
----------------------------------------------------------*/
void get_network_boxes(float* predictions, int netw, int neth, int GRID, int* masks, 
                       float* anchors, int &box_off, detection* dets)
{
	int lw = GRID;
	int lh = GRID;
	int nboxes = GRID * GRID * nanchor;
	int LISTSIZE = 1 + 4 + nclasses;
	//yolov5 output排列格式: box顺序为先grid再anchor
	//1个anchor: 7*7*x+7*7*y+7*7*w+7*7*w+7*7*obj+7*7*classes1+7*7*classes2, 共3个anchor
	
    //dets将outpus重排列,dets[i]为第i个框, det的index顺序为先anchor再grid
	for (int i = 0; i < lw * lh; i++) {
		int row = i / lw;
		int col = i % lw;
		for (int n = 0; n < nanchor; n++) {
//			int box_loc = n * lw * lh + i;
//			int box_index = n * lw * lh * LISTSIZE + i;  //第box_loc个box (x,y,w,h,...)起始位置, ywh索引只要依次加上lw*lh，i是相对于某一参数起始点的偏移量
			int box_index = (n * lw * lh + i) * LISTSIZE; //第box_loc个box的参数是 之后的7个数字
//			int obj_index = box_index + 4 * lw * lh;
			int obj_index = box_index + 4;
			float objectness = predictions[obj_index];
			if (objectness < OBJ_THRESH)
				continue;
			dets[box_off].objectness = objectness;
			// dets[box_off].classes = nclasses;
			dets[box_off].bbox = get_yolo_box(predictions, anchors, masks[n], box_index, col, row, lw, lh, netw, neth, lw * lh);
			for (int j = 0; j < nclasses; j++) {
				int class_index = box_index + (5 + j);
				float prob = objectness * predictions[class_index];
				dets[box_off].prob[j] = prob;
			}
			++box_off;
		}
	}
}

/*---------------------------------------------------------
	Yolov5 输出结果解码3
	对网络中3种尺寸的预测结果分别进行解码
----------------------------------------------------------*/
int outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets) {
	float* output_0 = (float*)rknn_outputs[0].buf;
	float* output_1 = (float*)rknn_outputs[1].buf;
	float* output_2 = (float*)rknn_outputs[2].buf;
	int masks_0[3] = { 0, 1, 2 };
	int masks_1[3] = { 3, 4, 5 };
	int masks_2[3] = { 6, 7, 8 };
	float anchors[18] = { 10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326 };
	//输出xywh均在0-1范围内
	int nboxes_valid = 0;
	get_network_boxes(output_0, net_width, net_height, GRID0, masks_0, anchors, nboxes_valid, dets);
	get_network_boxes(output_1, net_width, net_height, GRID1, masks_1, anchors, nboxes_valid, dets);
	get_network_boxes(output_2, net_width, net_height, GRID2, masks_2, anchors, nboxes_valid, dets);
	return nboxes_valid;
}



#include <unistd.h>

#include "common.h"
#include "mytime.h"
#include "decode.h"
#include "detect.h"
#include "videoio.h"

using namespace std;

extern bool bReading;
extern bool bDetecting;     // 目标检测进程状态
// 开始 结束时间 ns
extern double start_time;
extern video_property video_probs; // 视频属性类
// 多线程控制相关
extern int idxOutputImage;            // next frame index to be output 保证queueDetOut_server序号正确
extern mutex mtxQueueInput;               
extern queue<input_image> queueInput;  // input queue
extern mutex mtxQueueDetOut;
extern queue<imageout_idx> queueDetOut;// Det output queue

int Yolo::detect_process(){
	
	queue<float> history_time;
	float sum_time = 0;
	int cost_time = 0; // rknn接口查询返回
	float npu_performance = 0.0;

	// Letter box resize
	float img_wh_ratio = (float)IMG_WIDTH / (float)IMG_HEIGHT;
	float input_wh_ratio = (float)NET_INPUTWIDTH / (float)NET_INPUTHEIGHT;
	float resize_scale=0;
	int resize_width;
	int resize_height;
	int h_pad;
	int w_pad;
	if (img_wh_ratio >= input_wh_ratio){
		//pad height dim
		resize_scale = (float)NET_INPUTWIDTH / (float)IMG_WIDTH;
		resize_width = NET_INPUTWIDTH;
		resize_height = (int)((float)IMG_HEIGHT * resize_scale);
		w_pad = 0;
		h_pad = (NET_INPUTHEIGHT - resize_height) / 2;
	}
	else{
		//pad width dim
		resize_scale = (float)NET_INPUTHEIGHT / (float)IMG_HEIGHT;
		resize_width = (int)((float)IMG_WIDTH * resize_scale);
		resize_height = NET_INPUTHEIGHT;
		w_pad = (NET_INPUTWIDTH - resize_width) / 2;
		h_pad = 0;
	}

	while (1)
	{
		// cout << "Entering detect process" << queueInput.size() << "\n";
		//Load image
		input_image input;
		mtxQueueInput.lock();
		//queueInput不为空则进入NPU_process
		if (queueInput.empty()) {
			// printf("waiting queueInput .........\n");
			mtxQueueInput.unlock();
			usleep(1000);
			// 如果queue处于空且 bReading不在可读取状态则销毁 跳出
			if (bReading){
				continue;
			} 
			else {
				break;
			}
		}
		else{
			// Get an image from input queue
			// cout << "已缓存的图片数: " << queueInput.size() << endl;
			input = queueInput.front();
			queueInput.pop();
			mtxQueueInput.unlock();
		}

		if(input.index == 0){
			start_time = what_time_is_it_now();
		} 
		
		detect_result_group_t detect_result_group;
		// detection interval to speed up
		if (input.index < this->det_interval || !(input.index % this->det_interval)) {
			double timeBeforeDetection = what_time_is_it_now();
			cost_time = inference(input.img_pad.data);
			if(cost_time == -1)
				printf("NPU inference Error");

			std::vector<float> out_scales;
			std::vector<int32_t> out_zps;
			for (int i = 0; i < _n_output; ++i) {
				out_scales.push_back(_output_attrs[i].scale);
				out_zps.push_back(_output_attrs[i].zp);
			}

			post_process_fp((float *)_output_buff[0], (float *)_output_buff[1], (float *)_output_buff[2],
		 				NET_INPUTHEIGHT, NET_INPUTWIDTH, 0, 0, resize_scale, BOX_THRESH, NMS_THRESH, &detect_result_group);

			double timeAfterDetection = what_time_is_it_now();

			cout << "--------Time cost in Detection: " << timeAfterDetection - timeBeforeDetection << "\n";
			
		}
		
		// cout << "post process done\n";
		detect_result_group.id = input.index;
		// double end_time = what_time_is_it_now();
		// cost_time = end_time - start_time;
		npu_performance = cal_NPU_performance(history_time, sum_time, cost_time / 1.0e3);

		while(detect_result_group.id != idxOutputImage){
			usleep(1000);
		}
		imageout_idx res_pair;
		res_pair.img = input.img_src;
		res_pair.dets = detect_result_group;
		mtxQueueDetOut.lock();
		queueDetOut.push(res_pair);
		mtxQueueDetOut.unlock();
		// printf("%f NPU(%d) performance : %f (%d)\n", what_time_is_it_now()/1000, _cpu_id, npu_performance, detect_result_group.id);
		// draw_image(input.img_src, post_do.scale, nms_res, nboxes_left, 0.3);
		idxOutputImage = idxOutputImage + 1;
		if(input.index == video_probs.Frame_cnt-1){
			break; // 不加也可 queueInput.empty() + breading可以跳出
		}
	}
	cout << "Detect is over." << endl;
	bDetecting = false;
    return 0;
}

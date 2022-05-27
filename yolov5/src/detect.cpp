#include "detect.h"


float Yolo::cal_NPU_performance(queue<float> &history_time, float &sum_time, float cost_time){
	// 统计NPU在最近一段时间内的速度
	if(history_time.size()<10){
		history_time.push(cost_time);
		sum_time += cost_time;
	}
	else if(history_time.size()==10){
		sum_time -= history_time.front();
		sum_time += cost_time;
		history_time.pop();
		history_time.push(cost_time);
	}
	else{
		printf("cal_NPU_performance Error\n");
		return -1;
	}
	return sum_time / history_time.size();
}

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
		//Load image
		input_image input;
		mtxQueueInput.lock();
		//queueInput不为空则进入NPU_process
		if (queueInput.empty()) {
			//printf("waiting queueInput .........\n", ret);
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
		cost_time = inference(input.img_pad.data);
		if(cost_time == -1){
			printf("NPU inference Error");
		}
		
		detect_result_group_t detect_result_group;
		std::vector<float> out_scales;
		std::vector<int32_t> out_zps;
		for (int i = 0; i < _n_output; ++i)
		{
			out_scales.push_back(_output_attrs[i].scale);
			out_zps.push_back(_output_attrs[i].zp);
		}
		// if valid nbox is few, cost time can be ignored.
		// 补边左上角对齐 因此 w_pad = h_pad = 0
		// double start_time = what_time_is_it_now();
		post_process_fp((float *)_output_buff[0], (float *)_output_buff[1], (float *)_output_buff[2],
		 				NET_INPUTHEIGHT, NET_INPUTWIDTH, h_pad, w_pad, resize_scale, BOX_THRESH, NMS_THRESH, &detect_result_group);
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
		mtxqueueDetOut.lock();
		queueDetOut.push(res_pair);
		printf("%f NPU(%d) performance : %f (%d)\n", what_time_is_it_now()/1000, _cpu_id, npu_performance, detect_result_group.id);
		// draw_image(input.img_src, post_do.scale, nms_res, nboxes_left, 0.3);
		idxOutputImage = idxOutputImage + 1;
		mtxqueueDetOut.unlock();
		if(input.index == video_probs.Frame_cnt-1){
			end_time = what_time_is_it_now();
			break; // 不加也可 queueInput.empty() + breading可以跳出
		}
	}
	bDetecting = false;
    return 0;
}

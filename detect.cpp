#include "detect.h"



/*--------------------------------------------------------
	rknn_fp类
----------------------------------------------------------*/
// 读取rknn模型输入/输出属性
void rknn_fp::dump_tensor_attr(rknn_tensor_attr* attr)
{
	printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
			"zp=%d, scale=%f\n",
			attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
			attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
			get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
	return;
}
// rknn_fp构造函数 NPU初始化
rknn_fp::rknn_fp(const char *model_path, int cpuid, rknn_core_mask core_mask)
{
	int ret = 0;
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind NPU process on CPU %d\n", cpuid);


	// Load model

	FILE *fp = fopen(model_path, "rb");
	if(fp == NULL) {
		printf("fopen %s fail!\n", model_path);
		exit(-1);
	}
	// 文件的长度(单位字节)
	fseek(fp, 0, SEEK_END);
	int model_len = ftell(fp);
	// 创建一个存储空间model且读入
	void *model = malloc(model_len);
	fseek(fp, 0, SEEK_SET);
	if(model_len != fread(model, 1, model_len, fp)) {
		printf("fread %s fail!\n", model_path);
		free(model);
		exit(-1);
	}
	
    // ret = rknn_init(&ctx, model, m odel_len, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
	ret = rknn_init(&ctx, model, model_len, 0, NULL);
	if(ret < 0)
	{
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }
	ret = rknn_set_core_mask(ctx, core_mask);
	if(ret < 0)
	{
        printf("set NPU core_mask fail! ret=%d\n", ret);
        exit(-1);
    }
	// rknn_sdk_version
	rknn_sdk_version version;
	ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
	sizeof(rknn_sdk_version));
	printf("api version: %s\n", version.api_version);
	printf("driver version: %s\n", version.drv_version);

    // rknn inputs
	printf("input tensors:\n");
	memset(_input_attrs, 0, _n_input * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < _n_input; i++) {
		_input_attrs[i].index = i;
		// query info
		ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(_input_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret < 0) {
			printf("rknn_init error! ret=%d\n", ret);
			exit(-1);
		}
		dump_tensor_attr(&_input_attrs[i]);
	}

	// Create input tensor memory
	rknn_tensor_type   input_type   = RKNN_TENSOR_UINT8; // default input type is int8 (normalize and quantize need compute in outside)
	rknn_tensor_format input_layout = RKNN_TENSOR_NHWC; // default fmt is NHWC, npu only support NHWC in zero copy mode
	_input_attrs[0].type = input_type;
	_input_attrs[0].fmt = input_layout;
	_input_mems[0] = rknn_create_mem(ctx, _input_attrs[0].size_with_stride);

	// rknn outputs
	printf("output tensors:\n");
	memset(_output_attrs, 0, _n_output * sizeof(rknn_tensor_attr));
	for (uint32_t i = 0; i < _n_output; i++) {
		_output_attrs[i].index = i;
		// query info
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(_output_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			printf("rknn_query fail! ret=%d\n", ret);
			exit(-1);
		}
		dump_tensor_attr(&_output_attrs[i]);
	}

	// Create output tensor memory
	for (uint32_t i = 0; i < _n_output; ++i) {
		// default output type is depend on model, this require float32 to compute top5
		// allocate float32 output tensor
		int output_size = _output_attrs[i].n_elems * sizeof(float);
		_output_mems[i]  = rknn_create_mem(ctx, output_size);
	}

	// Set input tensor memory
	ret = rknn_set_io_mem(ctx, _input_mems[0], &_input_attrs[0]);
	if (ret < 0) {
		printf("rknn_set_io_mem fail! ret=%d\n", ret);
		exit(-1);
	}

	// Set output tensor memory
	for (uint32_t i = 0; i < _n_output; ++i) {
		// default output type is depend on model, this require float32 to compute top5
		_output_attrs[i].type = RKNN_TENSOR_FLOAT32;
		// set output memory and attribute
		ret = rknn_set_io_mem(ctx, _output_mems[i], &_output_attrs[i]);
		if (ret < 0) {
			printf("rknn_set_io_mem fail! ret=%d\n", ret);
			exit(-1);
		}
	}
}

rknn_fp::~rknn_fp(){
    rknn_destroy(ctx);
}

int rknn_fp::detect(cv::Mat img){
	// Rknn调用函数 返回推理时间
    int ret;

    // inputs[0].buf = img.data;
	int width  = _input_attrs[0].dims[2];
	memcpy(_input_mems[0]->virt_addr, img.data, width*_input_attrs[0].dims[1]*_input_attrs[0].dims[3]);
	// if(img.data) free(img.data);
	unsigned char * buff = (unsigned char *)_input_mems[0]->virt_addr;

    // rknn inference
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
	// query1: inference time
	rknn_perf_run perf_run;
	ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run,sizeof(perf_run));
	// printf("RKNN_QUERY_PERF_RUN: inference time %d\n", perf_run.run_duration);
	// query2: inference time per layer
	// rknn_perf_detail perf_detail;
	// ret = rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
	// printf("%s \n", perf_detail.perf_data);

    // rknn outputs get
	for(int i=0;i<_n_output;i++){
		_output_buff[i] = (float*)_output_mems[i]->virt_addr;
	}

    return perf_run.run_duration;
}

float cal_NPU_performance(queue<float> &history_time, float &sum_time, float cost_time){
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

int detect_process(const char *model_path, int cpuid, rknn_core_mask core_mask){
	
	rknn_fp detect_fp(model_path, cpuid, core_mask);// npu初始化
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
		h_pad = (NET_INPUTHEIGHT - resize_height)/ 2;
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
		cost_time = detect_fp.detect(input.img_pad);
		if(cost_time == -1){
			printf("NPU inference Error");
		}
		
		detect_result_group_t detect_result_group;
		std::vector<float> out_scales;
		std::vector<int32_t> out_zps;
		for (int i = 0; i < detect_fp._n_output; ++i)
		{
			out_scales.push_back(detect_fp._output_attrs[i].scale);
			out_zps.push_back(detect_fp._output_attrs[i].zp);
		}
		// if valid nbox is few, cost time can be ignored.
		// 补边左上角对齐 因此 w_pad = h_pad = 0
		// double start_time = what_time_is_it_now();
		post_process_fp((float *)detect_fp._output_buff[0], (float *)detect_fp._output_buff[1], (float *)detect_fp._output_buff[2],
		 				NET_INPUTHEIGHT, NET_INPUTWIDTH, 0, 0, resize_scale, BOX_THRESH, NMS_THRESH, &detect_result_group);
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
		printf("%f NPU(%d) performance : %f (%d)\n", what_time_is_it_now()/1000, cpuid, npu_performance, detect_result_group.id);
		// draw_image(input.img_src, detect_fp.post_do.scale, nms_res, nboxes_left, 0.3);
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

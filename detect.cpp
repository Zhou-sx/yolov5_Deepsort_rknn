#include "detect.h"

preprocess::preprocess(){
	scale = 1.0;
}

void preprocess::set_size(int height, int width){
    input_height = height;
    input_width = width;
}
void preprocess::resize(cv::Mat &img, cv::Mat &_img)
{
    scale = get_max_scale(img);
    // cout << scale << endl;
    int img_width_new = img.cols * scale;
    int img_height_new = img.rows * scale;
    // cout << img_width_new << " " << img_height_new << endl;
    cv::resize(img, _img, cv::Size(img_width_new, img_height_new), (0, 0), (0, 0), cv::INTER_LINEAR);
    cv::copyMakeBorder(_img, _img, 0, input_height - img_height_new, 0, input_width - img_width_new, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
//		imwrite("./border.jpg", _img);
}

// 在不丢失原图比例的同时，尽可能的伸缩；同时为了保证检测效果，只允许缩放，不允许放大。
float preprocess::get_max_scale(cv::Mat &img)
{
    float scale = min((float)input_width / img.cols, (float)input_height/img.rows);
    if(scale > 1) return 1;
    else return scale;
}


/*--------------------------------------------------------
	rknn_fp类
----------------------------------------------------------*/
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
	
    // ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
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
	inputs[0].index = 0;
	inputs[0].size = NET_INPUTWIDTH * NET_INPUTHEIGHT * NET_INPUTCHANNEL;
	inputs[0].pass_through = false;//需要type和fmt
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
    

	// rknn outputs
	outputs[0].want_float = true;
	outputs[0].is_prealloc = false;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = false;
	outputs[2].want_float = true;
	outputs[2].is_prealloc = false;

	// rknn outputs_attr
	outputs_attr[0].index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
	if(ret < 0) {
		printf("rknn_query fail! ret=%d\n", ret);
		exit(-1);
	}

    // 预处理类 resize至指定尺寸
	post_do.set_size(NET_INPUTHEIGHT, NET_INPUTWIDTH);
}

rknn_fp::~rknn_fp(){
    rknn_destroy(ctx);
}

int rknn_fp::detect(cv::Mat src){
	// Rknn调用函数 返回推理时间
    int ret;

    cv::Mat img;
    post_do.resize(src, img);
    inputs[0].buf = img.data;

    ret = rknn_inputs_set(ctx, 1, inputs);
    
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

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
    ret = rknn_outputs_get(ctx, 3, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
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

	while (1)
	{
		//Load image
		pair<int, cv::Mat> pairIndexImage;
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
			//Get an image from input queue
			// cout << "已缓存的图片数: " << queueInput.size() << endl;
			pairIndexImage = queueInput.front();
			queueInput.pop();
			mtxQueueInput.unlock();
		}

		cost_time = detect_fp.detect(pairIndexImage.second);
		if(cost_time == -1){
			printf("NPU inference Error");
		}

		// detection* dets=(detection*) calloc(nboxes_total,sizeof(detection));
		// int nboxes_valid = outputs_transform(detect_fp.outputs, NET_INPUTHEIGHT, NET_INPUTHEIGHT, dets);
		// detection* nms_res=(detection*) calloc(nboxes_total,sizeof(detection));
		// int nboxes_left = nms_sort(dets, nms_res, nboxes_valid, nclasses); 
		det_res detect_res;
		detect_res.idx = pairIndexImage.first;
		// detect_res.nboxes_left = nboxes_left;
		detect_res.nboxes_left = 0;

		npu_performance = cal_NPU_performance(history_time, sum_time, cost_time / 1.0e3);

		while(queueDetOut.size() > 300 || detect_res.idx != idxOutputImage){
			usleep(1000);
		}
		mtxqueueDetOut.lock();
		queueDetOut.push(detect_res);
		printf("%f NPU(%d) performance : %f (%d)\n", what_time_is_it_now()/1000, cpuid, npu_performance, detect_res.idx);
		idxOutputImage = idxOutputImage + 1;
		mtxqueueDetOut.unlock();
		
		// delete dets; //释放空间
		// delete nms_res;
	}
    return 0;
}

/*---------------------------------------------------------
	绘制预测框
----------------------------------------------------------*/
string labels[2]={"person", "vehicle"};
cv::Scalar colorArray[2]={
	cv::Scalar(139,0,0,255),
	cv::Scalar(139,0,139,255),
};
int draw_image(cv::Mat img,float scale,detection* dets,int total,float thresh)
{
	//::cvtColor(img, img, cv::COLOR_RGB2BGR);
	for(int i=0;i<total;i++){
		char labelstr[4096]={0};
		int class_=-1;
		int topclass=-1;
		float topclass_score=0;
		if(dets[i].objectness==0) continue;
		for(int j=0;j<nclasses;j++){
			if(dets[i].prob[j]>thresh){
				if(topclass_score<dets[i].prob[j]){
					topclass_score=dets[i].prob[j];
					topclass=j;
				}
				if(class_<0){
					strcat(labelstr,labels[j].data());
					class_=j;
				}
				else{
					strcat(labelstr,",");
					strcat(labelstr,labels[j].data());
				}
			}
		}
		//如果class>0说明框中有物体,需画框
		if(class_>=0){
			cv::Rect_<float> b=dets[i].bbox;
			//计算坐标 先根据缩放后的图计算绝对坐标 然后除以scale缩放到原来的图
			//又因为原点重合 因此缩放后的结果就是原结果
			int x1 = b.x * NET_INPUTWIDTH / scale;
			int x2= x1 + b.width * NET_INPUTWIDTH / scale + 0.5;
			int y1= b.y * NET_INPUTWIDTH / scale;
			int y2= y1 + b.height * NET_INPUTHEIGHT / scale + 0.5;

            if(x1  < 0) x1  = 0;
            if(x2> img.cols-1) x2 = img.cols-1;
            if(y1 < 0) y1 = 0;
            if(y2 > img.rows-1) y2 = img.rows-1;
			//std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[class_%10], 3);
            putText(img, labelstr, cv::Point(x1, y1 - 12), 1, 2, cv::Scalar(0, 255, 0, 255));
            }
		}
		// imwrite("./display.jpg", img);
	return 0;
}

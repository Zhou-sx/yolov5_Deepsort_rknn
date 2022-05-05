#include "videoio.h"


preprocess::preprocess(){
	input_height = NET_INPUTHEIGHT;
	input_width = NET_INPUTWIDTH;
	scale = get_max_scale(IMG_WIDTH, IMG_HEIGHT, NET_INPUTWIDTH, NET_INPUTHEIGHT);
}
void preprocess::resize(cv::Mat &img, cv::Mat &_img)
{
    int img_width_new = img.cols * scale;
    int img_height_new = img.rows * scale;
    // cout << img_width_new << " " << img_height_new << endl;
    cv::resize(img, _img, cv::Size(img_width_new, img_height_new), (0, 0), (0, 0), cv::INTER_LINEAR);
    cv::copyMakeBorder(_img, _img, 0, input_height - img_height_new, 0, input_width - img_width_new, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	// imwrite("./border.jpg", _img);
	return;
}


/*---------------------------------------------------------
	读视频
	video_name: 视频路径
	cpuid:		绑定到某核
----------------------------------------------------------*/
void videoRead(const char *video_name, int cpuid) 
{
	// int initialization_finished = 1;
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoTransClient process to CPU %d\n", cpuid); 

	cv::VideoCapture video;
	if (!video.open(video_name)) {
		cout << "Fail to open " << video_name << endl;
		return;
	}

	video_probs.Frame_cnt = video.get(CV_CAP_PROP_FRAME_COUNT);
    video_probs.Fps = video.get(CV_CAP_PROP_FPS);
    video_probs.Video_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    video_probs.Video_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    video_probs.Video_fourcc = video.get(CV_CAP_PROP_FOURCC);
	
	preprocess post_do;

	bReading = true;//读写状态标记
	while (1) 
	{  
		cv::Mat img_src;
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		if (!bReading || !video.read(img_src) || idxInputImage >= video_probs.Frame_cnt) {
			// cout << "read video stream failed! Maybe to the end!" << endl;
			// video.set(CV_CAP_PROP_POS_FRAMES, 0);
			// continue;
			video.release();
			break;
		}
		// resize 方法
		cv::Mat img_pad;
		post_do.resize(img_src, img_pad);
		// cv::Mat img_pad(IMG_PAD, IMG_PAD, CV_8UC3);
		// memcpy(img_pad.data, img_src.data, IMG_WIDTH*IMG_HEIGHT*IMG_CHANNEL);

		mtxQueueInput.lock();
		queueInput.push(input_image(idxInputImage, img_src, img_pad));
		mtxQueueInput.unlock();
		idxInputImage++;
	}
	bReading = false;
	cout << "VideoRead is over." << endl;
}

 /*
	预处理的缩放比例
	在不丢失原图比例的同时，尽可能的伸缩；同时为了保证检测效果，只允许缩放，不允许放大。
*/
float get_max_scale(int input_width, int input_height, int net_width, int net_height)
{
    float scale = min((float)net_width / input_width, (float)net_height /input_height);
    if(scale > 1) return 1;
    else return scale;
}

// 写视频
void videoWrite(const char* save_path,int cpuid) 
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoWrite process to CPU %d\n", cpuid); 

	cv::VideoWriter vid_writer;
    while(1)
    {
       if(queueInput.size() > 0)
       {
            // cout << video_probs.Video_width << " " << video_probs.Video_height << endl;
            vid_writer  = cv::VideoWriter(save_path, video_probs.Video_fourcc, video_probs.Fps, 
										  cv::Size(video_probs.Video_width, video_probs.Video_height));
            break;
       }
    }

	while (1) 
	{  
		cv::Mat img;

		// 如果queueDetOut存在元素 就尝试写
		if (queueDetOut.size() > 0) {
			mtxqueueDetOut.lock();
			imageout_idx res_pair = queueDetOut.front();
			queueDetOut.pop();
			mtxqueueDetOut.unlock();
			draw_image(res_pair.img, res_pair.dets);
			vid_writer.write(img); // Save-video
		}
		// 最后一帧检测结束 bWriting置为false 此时如果queueOutput仍存在元素 继续写
		else if(!bDetecting){
			vid_writer.release();
			break;
		}
	}
	cout << "VideoWrite is over." << endl;
}

/*---------------------------------------------------------
	绘制预测框
----------------------------------------------------------*/
string labels[2]={"person", "vehicle"};
cv::Scalar colorArray[2]={
	cv::Scalar(139,0,0,255),
	cv::Scalar(139,0,139,255),
};
int draw_image(cv::Mat img,detect_result_group_t detect_result_group)
{
	char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
	imwrite("./display.jpg", img);
	return 0;
}
#include "videoio.h"

/*---------------------------------------------------------
	传视频客户端
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
	
	bReading = true;//读写状态标记
	while (1) 
	{  
		cv::Mat img, img_gray;
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		if (!bReading || !video.read(img) || idxInputImage >= video_probs.Frame_cnt) {
			// cout << "read video stream failed! Maybe to the end!" << endl;
			// video.set(CV_CAP_PROP_POS_FRAMES, 0);
			// continue;
			video.release();
			break;
		}
		mtxQueueInput.lock();
		queueInput.push(make_pair(idxInputImage, img));
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
vector<float> get_max_scale(int input_width, int input_height, int net_width, int net_height)
{
    vector<float> scale = {  (float)net_width / input_width,
							 (float)net_height /input_height
					 	  };
	return scale;
}

// 写视频
void videoWrite(const char* save_path,int cpuid) 
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);
	vector<float> scale = get_max_scale(IMG_WIDTH, IMG_HEIGHT, NET_INPUTWIDTH, NET_INPUTHEIGHT); // 预处理缩放比例

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind VideoCapture process to CPU %d\n", cpuid); 

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
		usleep(10000);
		cv::Mat img;

		// 如果queueDetOut存在元素 就尝试写
		if (queueDetOut.size() > 0) {
			mtxqueueDetOut.lock();
			det_res detect_res = queueDetOut.front();
			queueDetOut.pop();
			mtxqueueDetOut.unlock();
			cv::Mat img = detect_res.img;
			draw_image(img, scale, detect_res.res, detect_res.nboxes_left, 0.3);
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
int draw_image(cv::Mat img,vector<float> scale,detection* dets,int total,float thresh)
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
			int x1 = b.x * NET_INPUTWIDTH / scale[0];
			int x2= x1 + b.width * NET_INPUTWIDTH / scale[0] + 0.5;
			int y1= b.y * NET_INPUTHEIGHT / scale[1];
			int y2= y1 + b.height * NET_INPUTHEIGHT / scale[1] + 0.5;

            if(x1  < 0) x1  = 0;
            if(x2> img.cols-1) x2 = img.cols-1;
            if(y1 < 0) y1 = 0;
            if(y2 > img.rows-1) y2 = img.rows-1;
			//std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[class_%10], 3);
            putText(img, labelstr, cv::Point(x1, y1 - 12), 1, 2, cv::Scalar(0, 255, 0, 255));
            }
		}
		imwrite("./display.jpg", img);
	return 0;
}
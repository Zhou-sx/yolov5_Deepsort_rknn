#include "videoio.h"
#include "main.h"

using namespace std;

extern video_property video_probs;
extern vector<cv::Mat> imagePool;
extern mutex mtxQueueInput;
extern queue<input_image> queueInput;  // input queue client
extern mutex mtxQueueDetOut;
extern queue<imageout_idx> queueDetOut;        // Det output queue
extern mutex mtxQueueOutput;
extern queue<imageout_idx> queueOutput;  // 目标追踪输出队列

extern bool add_head;
extern bool bReading;      // flag of input
extern bool bDetecting;    // 目标检测进程状态
extern bool bTracking;
extern int idxInputImage;  // image index of input video

preprocess::preprocess(){
	input_height = NET_INPUTHEIGHT;
	input_width = NET_INPUTWIDTH;
	input_channel = NET_INPUTCHANNEL;
}

void preprocess::resize(cv::Mat &img, cv::Mat &_img)
{
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
	int img_width = img.cols;
    int img_height = img.rows;
    void *resize_buf = malloc(input_height * input_width * input_channel);

	src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, input_width, input_height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);

	IM_STATUS STATUS = imresize(src, dst);
    if (ret != IM_STATUS_NOERROR)
    {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
	_img = cv::Mat(cv::Size(input_width, input_height), CV_8UC3, resize_buf);

	// cv::imwrite("resize_input.jpg", _img);
}

/*---------------------------------------------------------
	读视频 缓存在imagePool
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
		if (!video.read(img_src)) {
			cout << "read video stream failed! Maybe to the end!" << endl;
			video.release();
			break;
		}
		imagePool.emplace_back(img_src);
	}
	cout << "VideoRead is over." << endl;
}

/*---------------------------------------------------------
	调整视频尺寸
	cpuid:		绑定到某核
----------------------------------------------------------*/
void videoResize(int cpuid){
	// int initialization_finished = 1;
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoTransClient process to CPU %d\n", cpuid);

	preprocess post_do;
	bReading = true;//读写状态标记
	while (1) 
	{  
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		if (!bReading || idxInputImage >= video_probs.Frame_cnt) {
			break;
		}
		cv::Mat img_src = imagePool[idxInputImage];
		cv::Mat img_pad;
		if (add_head){
			// adaptive head
			img_pad = cv::Mat(IMG_PAD, IMG_PAD, CV_8UC3);
			memcpy(img_pad.data, img_src.data, IMG_WIDTH*IMG_HEIGHT*IMG_CHANNEL);
		}
		else{
			// rga resize
			post_do.resize(img_src, img_pad);
		}

		mtxQueueInput.lock();
		queueInput.push(input_image(idxInputImage, img_src, img_pad));
		mtxQueueInput.unlock();
		idxInputImage++;
	}
	bReading = false;
	cout << "VideoResize is over." << endl;
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
		// queueOutput 就尝试写
		if (queueOutput.size() > 0) {
			mtxQueueOutput.lock();
 			imageout_idx res_pair = queueOutput.front();
			queueOutput.pop();
			mtxQueueOutput.unlock();
			draw_image(res_pair.img, res_pair.dets);
			vid_writer.write(res_pair.img); // Save-video
		}
		// 最后一帧检测/追踪结束 bWriting置为false 此时如果queueOutput仍存在元素 继续写
		else if(!bTracking){
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
int draw_image(cv::Mat &img,detect_result_group_t detect_result_group)
{
	char text[256];
    for (auto det_result : detect_result_group.results)
    {
        // sprintf(text, "%s %.1f%%", det_result.name, det_result.confidence * 100);
		sprintf(text, "ID:%d", (int)det_result.trackID);
        int x1 = det_result.x1;
        int y1 = det_result.y1;
        int x2 = det_result.x2;
        int y2 = det_result.y2;
		int class_id = det_result.classID;
        rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[class_id%10], 3);
        putText(img, text, cv::Point(x1, y1 - 12), 1, 2, cv::Scalar(0, 255, 0, 255));
    }
	// imwrite("./display.jpg", img);
	return 0;
}
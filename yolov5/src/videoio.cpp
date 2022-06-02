#include "videoio.h"
#include "resize.h"
#include "common.h"

using namespace std;

extern video_property video_probs;
extern vector<cv::Mat> imagePool;
extern mutex mtxQueueInput;
extern queue<input_image> queueInput;  // input queue client
extern mutex mtxQueueDetOut;
extern queue<detect_result_group_t> queueDetOut;        // Det output queue
extern mutex mtxQueueOutput;
extern queue<track_result_group_t> queueOutput;  // 目标追踪输出队列

extern bool add_head;
extern bool bReading;      // flag of input
extern bool bDetecting;    // 目标检测进程状态
extern bool bTracking;
extern int idxInputImage;  // image index of input video



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

	PreResize pre_do(NET_INPUTHEIGHT, NET_INPUTWIDTH, NET_INPUTCHANNEL);
	bReading = true;//读写状态标记
	while (1) 
	{  
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		if (!bReading || idxInputImage >= video_probs.Frame_cnt) {
			break;
		}
		cv::Mat img_src = imagePool[idxInputImage];
		cv::Mat img_pad = cv::Mat(IMG_PAD, IMG_PAD, CV_8UC3);
		memcpy(img_pad.data, img_src.data, IMG_WIDTH*IMG_HEIGHT*IMG_CHANNEL);
		if (add_head){
			// adaptive head
		}
		else{
			// rga resize
			pre_do.resize(img_pad, img_pad);
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
	fx = 1 沿x轴缩放
	fy = 1 沿y轴缩放
*/
void get_max_scale(int input_width, int input_height, int net_width, int net_height, double &fx, double &fy)
{
    double img_wh_ratio = (double)input_width / (double)input_height;
	double input_wh_ratio = (double)net_width / (double)net_height;
	if (img_wh_ratio >= input_wh_ratio){
		// 缩放相同倍数 w 先到达边界
		fx = (double)net_width / input_width;
		fy = (double)net_width / input_width;
	}
	else{
		fx = (double)net_height / input_height;
		fy = (double)net_height / input_height;
	}
	return;
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
 			track_result_group_t res_pair = queueOutput.front();
			queueOutput.pop();
			mtxQueueOutput.unlock();
			draw_image(res_pair.img, res_pair.output_stracks);
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

cv::Scalar get_color(int idx)
{
	idx += 3;
	return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}

int draw_image(cv::Mat &img, vector<STrack> output_stracks)
{
	char text[256];
    for (auto output_strack : output_stracks)
    {
		sprintf(text, "ID:%d", (int)output_strack.track_id);
		vector<float> tlwh = output_strack.tlwh;
		cv::Scalar scalar = get_color(output_strack.track_id);
        rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), scalar, 3);
        putText(img, text, cv::Point(tlwh[0], tlwh[1] - 5), 1, 2, cv::Scalar(0, 255, 0, 255));
    }
	// imwrite("./display.jpg", img);
	return 0;
}
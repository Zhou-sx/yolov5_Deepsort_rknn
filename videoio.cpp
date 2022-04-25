#include "videoio.h"

video_property video_probs;
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

// 写视频
void videoWrite(const char* save_path,int cpuid) 
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

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
		usleep(10);
		cv::Mat img;

		// 如果queueOutput存在元素 就一直写
		if (queueOutput.size() > 0) {
			mtxQueueOutput.lock();
			img = queueOutput.front();
			queueOutput.pop();
			mtxQueueOutput.unlock();

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
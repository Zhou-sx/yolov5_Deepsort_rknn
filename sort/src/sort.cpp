#include <vector>
#include <unistd.h>

#include "sort.h"


struct video_property;
extern queue<detect_result_group_t> queueDetOut;  // output queue
extern queue<track_result_group_t> queueOutput;  // output queue 目标追踪输出队列
extern video_property video_probs;       // 视频属性类
extern bool bDetecting;                  // 目标检测进程状态
extern bool bTracking;                   // 目标追踪进程状态               
extern double end_time;                  // 整个视频追踪结束


Sort::Sort(int){
	KalmanTracker::kf_count = 0;
	get_color();
}

Sort::~Sort(){

}

void Sort::get_color(){
	// 随机生成颜色
	cv::RNG rng(0xFFFFFFFF); //RNG类是opencv里C++的随机数产生器
	for (int i = 0; i < CNUM; i++)
		rng.fill(_randColor[i], RNG::UNIFORM, 0, 256);  // randColor类成员变量
}


/*---------------------------------------------------------
	SORT 目标追踪
----------------------------------------------------------*/
int Sort::track_process()
{
    while (1) 
	{
		if (queueDetOut.empty()) {
            if(!bDetecting){
                // queueDetOut为空并且 bDetecting标志已经全部检测完成 说明追踪完毕了
                end_time = what_time_is_it_now();
                bTracking = false;
                break;
            }
			usleep(1000);
            continue;
		}

		std::vector<DetectBox> detFrameData = queueDetOut.front().results; // 当前帧的所有检测结果 
		if (trackers.size() == 0) // 追踪器是空(第一帧 或者 跟踪目标丢失)
		{
			//用第一帧的检测结果初始化跟踪器
			for (unsigned int i = 0; i < detFrameData.size(); i++)
			{
				KalmanTracker trk = KalmanTracker(detFrameData[i].bbox);
				trackers.push_back(trk);
			}
			// continue;
		}
		else
		{
			//3.1 预测已有跟踪器在当前帧的bb
			std::vector<cv::Rect_<float>> predictedBoxes;//预测bb
			predictedBoxes.clear();
			for (auto it = trackers.begin(); it != trackers.end();)
			{
				cv::Rect_<float> pBox = (*it).predict();
				if (pBox.x >= 0 && pBox.y >= 0)
				{
					predictedBoxes.push_back(pBox);
					it++;
				}
				else
				{
					it = trackers.erase(it);//bb不合理的tracker会被清除
					//cerr << "Box invalid at frame: " << idxShowImage << endl;
				}
			}
			//cout << "3.1 over" << endl;
			// 3.2. 使用匈牙利算法进行匹配
			std::vector<std::vector<double>> iouMatrix;
			iouMatrix.clear();
			unsigned int trkNum = 0;
			unsigned int detNum = 0;
			trkNum = predictedBoxes.size(); //由上一帧预测出来的结果
			detNum = detFrameData.size(); //当前帧的所有检测结果的 视作传感器的结果
			iouMatrix.resize(trkNum, std::vector<double>(detNum, 0)); //提前开好空间 避免频繁重定位
			for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
			{
				for (unsigned int j = 0; j < detNum; j++)
				{
					// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
					iouMatrix[i][j] = 1 - box_iou(predictedBoxes[i], detFrameData[j].bbox);
				}
			}

			HungarianAlgorithm HungAlgo;
			std::vector<int> assignment; //匹配结果 给每一个trk找一个det
			assignment.clear();
			if(trkNum!=0)
			{
				HungAlgo.Solve(iouMatrix, assignment);//匈牙利算法核心
			}
				// find matches, unmatched_detections and unmatched_predictions
			set<int> unmatchedDetections; // 没有被配对的检测框 说明有新目标出现
			set<int> unmatchedTrajectories; // 没有被配对的追踪器 说明有目标消失
			set<int> allItems;
			set<int> matchedItems;
			std::vector<cv::Point> matchedPairs; // 最终配对结果 trk-det
			unmatchedTrajectories.clear();
			unmatchedDetections.clear();
			allItems.clear();
			matchedItems.clear();
			if (detNum > trkNum) // 检测框的数量 大于 现存追踪器的数量
			{
				for (unsigned int n = 0; n < detNum; n++)
					allItems.insert(n);
				for (unsigned int i = 0; i < trkNum; ++i)
					matchedItems.insert(assignment[i]);
				set_difference( allItems.begin(), allItems.end(),
								matchedItems.begin(), matchedItems.end(),
								insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
					/*
						set_difference, 求集合1与集合2的差集 即可以找到没有被追踪的 det
						参数：第一个集合的开始位置，第一个集合的结束位置，
							第二个参数的开始位置，第二个参数的结束位置，
							结果集合的插入迭代器。
					*/
			}
			else if (detNum < trkNum) // 检测框的数量 小于 现存追踪器的数量; 追踪目标暂时消失
			{
				for (unsigned int i = 0; i < trkNum; ++i)
					if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
						unmatchedTrajectories.insert(i);
			}
			else //两者数量相等不做操作
				;

			// 过滤掉低IoU的匹配
			matchedPairs.clear();
			for (unsigned int i = 0; i < trkNum; ++i)
			{
				if (assignment[i] == -1) 
					continue;
				if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
				{
					unmatchedTrajectories.insert(i);
					unmatchedDetections.insert(assignment[i]);
				}
				else
					matchedPairs.push_back(cv::Point(i, assignment[i])); // 符合条件 成功配对
			}
			//cout << "3.2 over" << endl;
			
			// 3.3. 更新跟踪器
			// update matched trackers with assigned detections.
			// each prediction is corresponding to a tracker
			int detIdx, trkIdx;
			for (unsigned int i = 0; i < matchedPairs.size(); i++)
			{
				trkIdx = matchedPairs[i].x;
				detIdx = matchedPairs[i].y;
				trackers[trkIdx].update(detFrameData[detIdx].bbox);
			}
			// 给未匹配到的检测框创建和初始化跟踪器
			// unmatchedTrajectories没有操作 所以有必要保存unmatchedTrajectories吗?(maybe not)
			for (auto umd : unmatchedDetections)
			{
				KalmanTracker tracker = KalmanTracker(detFrameData[umd].bbox);
				trackers.push_back(tracker);
			}
		}
		// 获得跟踪器输出
		int max_age = 3;
		int min_hits = 3;
		//m_time_since_update：tracker距离上次匹配成功间隔的帧数
		//m_hit_streak：tracker连续匹配成功的帧数
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			// 输出条件：当前帧和前面2帧（连续3帧）匹配成功才记录
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits))//河狸
			{
				DetectBox res;
				res.bbox = (*it).get_state();
				res.trackID = (*it).m_id + 1;
				res.x1= res.bbox.x;
				res.y1 = res.bbox.y;
				res.x2 = res.bbox.x + res.bbox.width;
				res.y2 = res.bbox.y + res.bbox.height;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;
			if (it != trackers.end() && (*it).m_time_since_update > max_age)//连续3帧还没匹配到，清除
				it = trackers.erase(it);
		}
		
		track_result_group_t track_result_group;
		track_result_group.img = queueDetOut.front().img;
		track_result_group.results = frameTrackingResult;
		printf("%d\n", frameTrackingResult.size());
		queueOutput.push(track_result_group);
		queueDetOut.pop();
	}
	return 0;
}

/*
	计算IoU
*/
double box_iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
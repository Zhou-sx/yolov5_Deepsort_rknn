# Yolov5_DeepSORT_rknn

****
# 改动: 

## 本仓库在原仓库的基础上:
1. 改善了边界框漂移, 完善了当图中没有目标等其他情形出现的bug, 增加了对cost matrix出现nan时的处理
2. 加入了隔帧检测的功能. 设置方法:
> 在`./yolov5/include/detect.h`中 将
> `const int det_interval = 1;`改成期望的数值, 例如3, 表示每隔3帧检测一次, 这样可以**显著提升速度**. 
> 同时, 也需要更改`./deepsort/include/deepsort.h`中`line 39`的`const int track_interval = 1; `, 数值要和检测的保持一致.
3. 加入Re-ID多线程的功能
> 如果您不希望使用多线程, 则在`./deepsort/src/deepsort.cpp`中`line 144`的`if (numOfDetections < 2)`
> 改成`if (true)`  


自己使用时, 除了更改OpenCV的路径外, 要在`./include/common.h`中修改`IMG_WIDTH, IMG_HEIGHT, IMG_PAD, OBJ_CLASS_NUM`
在`./yolov5/src/decode.cpp`中修改`LABEL_NALE_TXT_PATH`.
****

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Yolov5_DeepSORT_rknn是基于瑞芯微Rockchip Neural Network(RKNN)开发的目标跟踪部署仓库，除了DeepSORT还支持SORT算法，可以根据不同的嵌入式平台选择合适的跟踪算法。本仓库中的DeepSORT在Rk3588上测试通过，SORT和ByteTrack应该在Rk3588和Rk3399Pro上都可运行。

下面是我们的演示视频 具有强烈的抖动，进一步提高目标检测模型精度并且在视频无抖动情况下追踪性能应该会很Nice。

<div align="center">
  <img src="https://github.com/Zhou-sx/yolov5_Deepsort_rknn/blob/deepsort/detect.gif" width="45%" />&emsp; &emsp;<img src="https://github.com/Zhou-sx/yolov5_Deepsort_rknn/blob/deepsort/deepsort.gif" width="45%" />
  <br/>
  <font size=5>Detect</font>
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <font size=5>DeepSORT</font>
  <br/>
</div>

<div align="center">
  <img src="https://github.com/Zhou-sx/yolov5_Deepsort_rknn/blob/deepsort/SORT.gif" width="45%" />&emsp; &emsp;<img src="https://github.com/Zhou-sx/yolov5_Deepsort_rknn/blob/deepsort/Bytetrack.gif" width="45%" />
  <br/>
  <font size=5>SORT</font>
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
	&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <font size=5>Bytetrack</font>
  <br/>
</div>

DeepSORT、SORT和ByteTrack已经上线，放在三个分支里！除了这三个算法之外，可能还会更新其他SOTA跟踪算法，多多关注~！

我是野生程序猿，如果在代码编写上存在不规范的情况，请多多见谅。

## 文档内容

- [文件目录结构描述](#文件目录结构描述)
- [安装及使用](#安装及使用)
- [数据说明](#数据说明)
- [性能测试](#性能测试)
- [参考仓库](#参考仓库)

## 文件目录结构描述

```
├── Readme.md                   // help
├── data						// 数据
├── model						// 模型
├── build
├── CMakeLists.txt			    // 编译Yolov5_DeepSORT
├── include						// 通用头文件
├── src
├── 3rdparty                    
│   ├── linrknn_api				// rknn   动态链接库
│   ├── rga		                // rga    动态链接库
│   ├── opencv		            // opencv 动态链接库(自行编译并在CmakeLists.txt中设置相应路径)
├── yolov5           			
│   └── include
│       └── decode.h            // 解码
│       └── detect.h            // 推理
│       └── videoio.h           // 视频IO
│   └── src
│       └── decode.cpp    
│       └── ...
├── deepsort
│   └── include
│       └── deepsort.h     		// class DeepSort
│       └── featuretensor.h     // Reid推理
│       └── ...
│   └── src
│       └── deepsort.cpp
│       └── ...
│   └── CMakeLists.txt			// 编译deepsort子模块

```

## 安装及使用

+ RKNN-Toolkit

  这个项目需要使用RKNN-Toolkit2(Rk3588)或者RKNN-Toolkit1(Rk3399Pro)，请确保librknnrt.so和rknn_server正常运行。可以先运行瑞芯微仓库中的Demo来测试。

    ```
    rknpu2
    https://github.com/rockchip-linux/rknpu2
    ```

+ opencv的编译安装

  可以选择直接在板子上编译，Rk3588编译速度很快，不到十分钟。
  也可以选择交叉编译，我使用的交叉编译工具链：gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu

    ```
    注意！！！
    请根据自己的OpenCV路径修改CMakeLists.txt文件

    搜索 set(OpenCV_DIR /home/linaro/workspace/opencv/lib/cmake/opencv4)
    将路径替换成你的OpenCVConfig.cmake所在的文件夹

    本项目中有两个CMakeLists.txt请同时修改！！！
    ```

+ DeepSort选用的模型是TorchReID中的osnet_x0_25 ，输入尺寸是256x512

  目前还没有针对于自己的数据集重新训练

  ```
  Torchreid
  https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
  ```

+ 项目编译与运行

  参考Cmake的使用，会在build目录下生成可执行文件。

## 数据说明

+ 使用的是红外车辆、行人数据集，自行拍摄的，暂时不公开。

+ 测试的视频存在严重的抖动，影响了跟踪性能，不代表DeepSORT的跟踪性能。

## 性能测试

目前只对模型速度进行了测试，首先送上瑞芯微官方的benchmark

+ 瑞芯微rknn_model_zoo

| platform（fps）          | yolov5s-relu | yolov5s-silu | yolov5m-relu | yolov5m-silu |
| ------------------------ | ------------ | ------------ | ------------ | ------------ |
| rk1808 - u8              | 35.24        | 26.41        | 16.27        | 12.60        |
| rv1109 - u8              | 19.58        | 13.33        | 8.11         | 5.45         |
| rv1126 - u8              | 27.54        | 19.29        | 11.69        | 7.86         |
| rk3566 - u8              | 15.16        | 10.60        | 8.65         | 6.61         |
| rk3588 - u8(single core) | 53.73        | 33.24        | 22.31        | 14.74        |

+ Ours

  DeepSORT的ReID网络单次推理耗时约3ms，但是由于每个检测框都需要推理一次网络故受目标个数影响很大。SORT和ByteTrack由于没有ReID网络，在目标不是很多的情况下跟踪时间约等于目标检测，但是ID切换现象会更明显更严重一些。
  
| platform（ms）          | yolov5s-relu | yolov5s-relu+Deepsort |yolov5s-relu+Sort   |yolov5s-relu+ByteTrack   |
| :-------------------------: | :--: | :--: | :--:  | :--: |
| rk3588 - u8(single core)    | 24 |   -  |   -   |   -  |
| rk3588 - u8(double core)    | 12 | 33.24(infulenced)| 12 | 12 |
| rk3399Pro - u8(single core) |   -  |   -  |   -   |   -  |

## 参考仓库

本项目参考了大量前人的优秀工作，在最后放上一些有用的仓库链接。

1. https://github.com/ultralytics/yolov5
2. https://github.com/airockchip/rknn_model_zoo
3. https://github.com/airockchip/librga
4. https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt
5. https://github.com/KaiyangZhou/deep-person-reid

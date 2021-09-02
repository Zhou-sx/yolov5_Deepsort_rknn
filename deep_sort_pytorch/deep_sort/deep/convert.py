import torch
import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.transforms as transforms
from model import Net

t7_path = './checkpoint/ckpt.t7'
pt_path = './checkpoint/ckpt.pt'
rknn_path = './checkpoint/ckpt.rknn'

def export_pytorch():
    net = Net(reid=True)
    state_dict = torch.load(t7_path, map_location=torch.device('cuda'))[
        'net_dict']
    net.load_state_dict(state_dict)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1,3,128,64))

    trace_model.save(pt_path)


def export_rknn():
    '''
    由.pt 模型转化得到 .rknn模型
    :return:
    '''
    model = pt_path
    input_size_list = [[3, 128, 64]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    # 这个模型 不要归一化 归一化之后推理结果完全不对
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], quantized_dtype='asymmetric_quantized-u8',
                optimization_level=1, batch_size=50, epochs=-1, target_platform=['rk1808'])
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=True, dataset='./quantity_dataset.txt')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # 量化精度分析
    # print('--> Accuracy analysis')
    # rknn.accuracy_analysis(inputs='./quantity_dataset.txt', target='rk1808',
    #                        output_dir='./accuracy_analysis')
    # print('done')
    # print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export .rknn failed!')
        exit(ret)
    print('done')

    return

class preprocess(object):
    """
        use Numpy instead of Torch.
    """
    def __init__(self):
        super(preprocess, self).__init__()
        self.size = (64, 128)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1 (delete)
            2. resize to (64, 128) as Market1501 dataset did
            3. normalize
            4. concatenate to a numpy array
        """
        def _resize(im, size):
            # 取消归一化
            return cv2.resize(im.astype(np.float32), size)
        # im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
        #     0) for im in im_crops], dim=0).float()
        _batch = []
        for im in im_crops:
            _im = _resize(im, self.size)
            for cn in range(3):
                _im[..., cn] = (_im[..., cn] - self.mean[cn]) / self.std[cn]
            if not _batch:
                _batch = _im
            else:
                np.concatenate(_batch, _im)
        return _batch


def inference_rknn():
    '''
    测试.rknn 的推理功能
    :return:
    '''
    # Create RKNN object
    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)

    # 输入图像 1
    img_raw = [cv2.imread("demo.jpg")]
    pre_process = preprocess()
    img = pre_process.preprocess(img_raw)
    # 输入图像 2
    # img = cv2.imread('demo2.jpg').transpose((2,1,0))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')

    outputs = rknn.inference(inputs=[img])
    print('done')

    rknn.release()

    return outputs


def inference_pt():
    mode = 1  # 0: .t7; 1: .pt
    net = Net(reid=True)
    if mode == 0:
        state_dict = torch.load(t7_path, map_location=torch.device('cuda'))[
            'net_dict']
        net.load_state_dict(state_dict)
    else:
        ftrace = torch.load(pt_path, map_location=torch.device('cuda'))
    net.eval()
    # 输入图像 1
    img_raw = [cv2.imread("demo.jpg")]
    pre_process = preprocess()
    img = pre_process.preprocess(img_raw)
    # 输入图像 2
    # img = [cv2.imread('demo2.jpg').transpose((2,1,0))]

    with torch.no_grad():
        if mode == 0:
            outputs = net(torch.Tensor(img)).numpy()
        else:
            outputs = ftrace.forward(img).numpy()
    return outputs


if __name__ == '__main__':
    # 导出.pt 模型
    # export_pytorch()
    # 导出.rknn 模型
    # export_rknn()
    # 测试推理
    res_rknn = inference_rknn()
    # res_pt = inference_pt()
    # cmp = res_rknn[0] - res_pt
    pass


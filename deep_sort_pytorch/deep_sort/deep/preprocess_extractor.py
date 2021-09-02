import numpy as np
import cv2


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

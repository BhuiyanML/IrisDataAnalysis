import numpy as np
import torch


def iou_score(output, target, need_sigmoid=True):
    smooth = 1e-10

    if torch.is_tensor(output):
        if need_sigmoid:
            output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    # print(output_.shape, target_.shape)

    intersection = np.sum((output_ & target_), axis=(1, 2))
    union = np.sum((output_ | target_), axis=(1, 2))

    iou_indiv = (intersection + smooth) / (union + smooth)
    iou = np.mean(iou_indiv)

    return iou

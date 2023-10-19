import os
import random
import numpy as np
import pandas as pd

import torch
from config import CFG


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def box_cxcywh_to_xyxy(x):
    # 中心点的x,y 以及宽高
    x_c, y_c, w, h = x.unbind(-1)
    # 返回的是左上，右下两个点的四个坐标
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def adjust_box_transform2origin(boxes:torch.Tensor,origin_shape):
    origin_h, origin_w, _ = origin_shape
    scale_fct = np.array([1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size,1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size],dtype='float32')

    if origin_w > origin_h:
        pad_top = int((CFG.img_size - origin_h/scale_fct[0])/2.0)
        boxes[:,[1, 3]] -= pad_top
        boxes *= scale_fct[0]
    else:
        pad_left = int((CFG.img_size - origin_w/scale_fct[1])/2.0)
        boxes[:,[0,2]] -= pad_left
        boxes *= scale_fct[1]
    return boxes
    

def tensor2np(img:torch.Tensor):
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1,2,0))
    return img


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=CFG.device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == CFG.pad_idx)

    return tgt_mask, tgt_padding_mask


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

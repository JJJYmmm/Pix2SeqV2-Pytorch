import cv2
import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('./')
from dataset.coco_object_detection import CoCoDetectionTest
from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, postprocess
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize
from utils import seed_everything

def parse_targets(img_id: int,
                    coco_targets: list,
                    w: int = None,
                    h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax] 
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        # x_max>x_min and y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = img_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["origin_size"] = [h,w] 

        return target

if __name__ == '__main__':
    seed_everything(42) 
    assert os.path.exists(CFG.coco_label_path), "json file {} dose not exist.".format(CFG.coco_label_path)
    with open(CFG.coco_label_path, 'r') as f:
        id2cls = json.load(f) # num = 90,exclude background(id=0)
        id2cls = {int(i) : cls_name for i, cls_name in id2cls.items()}
    cls2id = {cls_name : i for i, cls_name in id2cls.items()}
    max_cls_id = max(id2cls.keys())
    print(max_cls_id+1)     

    anno_path = os.path.join(CFG.dir_root, "coco_annotations","instances_val2017.json")
    img_root = os.path.join(CFG.dir_root, "val2017")
    coco = COCO(anno_path)
    ids = list(sorted(coco.imgs.keys()))

    imgid = ids[111]
    print(imgid)
    
    ann_ids = coco.getAnnIds(imgIds=imgid)
    coco_target = coco.loadAnns(ann_ids)

    path = coco.loadImgs(imgid)[0]['file_name']
    img = Image.open(os.path.join(img_root, path)).convert("RGB")
    img = np.asarray(img)

    h, w,_ = img.shape
    target = parse_targets(imgid, coco_target, w, h)

    boxes = target["boxes"]
    origin_h, origin_w = target['origin_size']
    scale_fct = np.array([1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size,1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size],dtype='float32')

    if origin_w > origin_h:
        boxes /= scale_fct[0]
        pad_top = int((CFG.img_size - origin_h/scale_fct[0])/2.0)
        boxes[:,[1, 3]] += pad_top
    else:
        boxes /= scale_fct[1]
        pad_left = int((CFG.img_size - origin_w/scale_fct[1])/2.0)
        boxes[:,[0,2]] += pad_left

    print(img.shape)
    img = A.Compose([
            A.LongestMaxSize(max_size=CFG.img_size, interpolation=1),
            A.PadIfNeeded(min_height=CFG.img_size, min_width=CFG.img_size, border_mode=0, value=(0,0,0)),
            ])(image=img)['image']
    print(img.shape)

    visualize(img, boxes, target["labels"].tolist(), id2cls, show=False)

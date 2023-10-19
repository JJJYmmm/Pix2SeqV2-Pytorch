import os
import copy

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from . import transforms

def get_transform_train(fixed_size,person_kps_info):
    return transforms.Compose([
        transforms.AffineTransform(scale=(2, 2), rotation=(-45, 45), fixed_size=(fixed_size,fixed_size)),
        transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ] 
    )
    
def get_transform_valid(fixed_size):
    return transforms.Compose([
        transforms.AffineTransform(scale=(2, 2), fixed_size=(fixed_size,fixed_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



class CocoKeypoint(data.Dataset):
    def __init__(self,
                 root,
                 dataset="train",
                 years="2017",
                 transforms=None,
                 fixed_size=(384, 384),
                 tokenizer=None):
        super().__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"person_keypoints_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, f"{dataset}{years}")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.coco = COCO(self.anno_path)
        img_ids = list(sorted(self.coco.imgs.keys()))

        det = self.coco

        self.valid_person_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = det.getAnnIds(imgIds=img_id)
            anns = det.loadAnns(ann_ids)
            for ann in anns:
                # only save person class
                if ann["category_id"] != 1:
                    print(f'warning: find not support id: {ann["category_id"]}, only support id: 1 (person)')
                    continue

                # skip objs without keypoints annotation
                if "keypoints" not in ann:
                    continue
                if max(ann["keypoints"]) == 0:
                    continue

                xmin, ymin, w, h = ann['bbox']
                # Use only valid bounding boxes
                if w > 0 and h > 0:
                    info = {
                        "box": [xmin, ymin, w, h],
                        "image_path": os.path.join(self.img_root, img_info["file_name"]),
                        "image_id": img_id,
                        "image_width": img_info['width'],
                        "image_height": img_info['height'],
                        "obj_origin_hw": [h, w],
                        "obj_index": obj_idx,
                        "score": ann["score"] if "score" in ann else 1.
                    }

                    keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                    visible = keypoints[:, 2]
                    keypoints = keypoints[:, :2]
                    info["keypoints"] = keypoints
                    info["visible"] = visible

                    self.valid_person_list.append(info)
                    obj_idx += 1

    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_person_list[idx])

        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)
        
        if self.tokenizer is not None:
            seqs, init_len = self.tokenizer.encode_keypoint(person_info)
            seqs = torch.LongTensor(seqs)
            return image, seqs, init_len

        return image, person_info

    def __len__(self):
        return len(self.valid_person_list)

def collate_fn(batch,max_len,pad_idx):
    image_batch, seq_batch, init_len_batch = [], [], []
    for image, seq, len in batch:
        image_batch.append(image)
        seq_batch.append(seq)
        init_len_batch.append(len)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                        seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch, init_len_batch

def get_loaders(dir_root,tokenizer, person_kps_info, fixed_size, batch_size, max_len, pad_idx, num_workers=2):

    train_ds = CocoKeypoint(root=dir_root,dataset='train',transforms=get_transform_train(fixed_size,person_kps_info), tokenizer=tokenizer)
    
    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = CocoKeypoint(root=dir_root,dataset='val',transforms=get_transform_valid(fixed_size), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader

if __name__ == '__main__':
    print('yes')
    coco = CocoKeypoint('/mnt/MSCOCO')
    print(coco[0])
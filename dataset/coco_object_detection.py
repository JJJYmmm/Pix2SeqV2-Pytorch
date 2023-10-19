import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from functools import partial
import albumentations as A
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from albumentations.pytorch import ToTensorV2

def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.LongestMaxSize(max_size=size, interpolation=1),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    # with pixel normalization
    # return A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.LongestMaxSize(max_size=size, interpolation=1),
    #     A.ToFloat(max_value=255),
    #     A.Normalize(max_pixel_value=1.0),
    #     A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
    #     ToTensorV2()
    #     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def get_transform_valid(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=1),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    # with pixel normalization 
    # return A.Compose([
    #     A.LongestMaxSize(max_size=size, interpolation=1),
    #     A.ToFloat(max_value=255),
    #     A.Normalize(max_pixel_value=1.0),
    #     A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
    #     ToTensorV2()
    #     ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_transform_test(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=1),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        A.Normalize(),
        ToTensorV2()
    ])



def _coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids


class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, dataset="train", transforms=None,tokenizer=None):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = "instances_{}2017.json".format(dataset)
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, "{}2017".format(dataset))
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "coco_annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = dataset
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.coco = COCO(self.anno_path)

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("coco91_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = _coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
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

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        img = np.asarray(img)

        h, w,_ = img.shape

        target = self.parse_targets(img_id, coco_target, w, h)

        labels = target['labels']
        bboxes = target["boxes"]        
        
        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        
        # jump empty sequence,important!!!
        if len(bboxes) == 0:
            idx = np.random.randint(0,len(self)-1)
            print(f'jump bad example img_id={img_id}')
            return self[idx]
         
        # img = torch.FloatTensor(img).permute(2,0,1) # replaced by ToTensorV2

        if self.tokenizer is not None:
            seqs = self.tokenizer.encode_box(labels, bboxes)
            seqs = torch.LongTensor(seqs)
            # init_len = 0, object detection and captioning have no prompts
            return img, seqs, 0

        # target['boxes'] = bboxes
        # target["labels"] = labels

        del target['boxes']
        del target["labels"]
        del target['area']
        del target['iscrowd']
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

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

def get_loaders(dir_root,tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2):

    train_ds = CocoDetection(root=dir_root,dataset='train',transforms=get_transform_train(img_size), tokenizer=tokenizer)
    
    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = CocoDetection(root=dir_root,dataset='val',transforms=get_transform_valid(img_size), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader

class CoCoDetectionTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        super(CoCoDetectionTest, self).__init__()
        self.img_paths = img_paths
        self.transforms = get_transform_test(size)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)[..., ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        # img = torch.FloatTensor(img).permute(2, 0, 1) # replaced by ToTensorV2

        return img

    def __len__(self):
        return len(self.img_paths)
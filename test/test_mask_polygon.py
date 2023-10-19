from pycocotools.coco import COCO
import cv2
import os,sys
import numpy as np
sys.path.append('../')

from dataset.transforms import AffineTransform
from visualize import visualize_mask
from dataset.coco_segmentation import get_transform_valid
from config import CFG

if __name__ == '__main__':
    coco = COCO('/mnt/MSCOCO/annotations/instances_val2017.json')
    ids = list(sorted(coco.imgs.keys()))
    img_id = 1000
    anns = coco.getAnnIds(imgIds=img_id)
    img_info = coco.loadImgs(ids=img_id)[0]
    img = cv2.imread(os.path.join('/mnt/MSCOCO/val2017',img_info['file_name']))
    cv2.imwrite('./test.png',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann = anns[3]
    ann_info = coco.loadAnns(ann)[0]
    segmentation = ann_info['segmentation']
    bbox = ann_info['bbox']
    print(bbox)
    
    polys = [np.array(poly,dtype=np.int32).reshape([-1,2]) for poly in segmentation]
    
    target = dict(box=bbox,segmentation=polys)
    transforms = AffineTransform(scale=(1.05, 1.05), fixed_size=(CFG.img_size,CFG.img_size))
    img, target = transforms(img, target)

    polys = target['segmentation']
    polys = [np.array(poly,dtype=np.int32) for poly in polys]
    print(polys)

    visualize_mask(img[...,::-1], polys)
    

    
    


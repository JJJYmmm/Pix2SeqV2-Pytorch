import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np

GT_COLOR = (0, 255, 0) # Green
PRED_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

# for line in coco keypoints
BODY_PARTS= [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7)
]

def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """Visualizes a single bounding box on the image"""
    bbox = [int(item) for item in bbox]
    x_min, y_min, x_max, y_max = bbox
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(text_height * 1.3)), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min+ int(text_height * 1.3)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_keypoint(img, keypoint_list,save_name='result'):
    img = img.copy()

    for keypoints in keypoint_list:
        for x, y in keypoints:
            if x == 0 and y == 0:
                continue
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0))
    
        for part in BODY_PARTS:
            keypoint_1 = keypoints[part[0] - 1] 
            x_1 = int(keypoint_1[0])
            y_1 = int(keypoint_1[1])
            if x_1 == 0 and y_1 == 0:
                continue
            keypoint_2 = keypoints[part[1] - 1]
            x_2 = int(keypoint_2[0])
            y_2 = int(keypoint_2[1])
            if x_2 == 0 and y_2 == 0:
                continue
            cv2.line(img, (x_1, y_1), (x_2, y_2), (0,255,0))

    cv2.imwrite(save_name+'.png', img)

def visualize_mask(img, polys, save_name='result'):
    '''
    img : read by cv2
    polys : [poly1,poly2,...polyn(point1,point2)] 
    poly size : [num_point,2]
    '''
    img = img.copy()
    mask = np.zeros((img.shape), dtype=np.uint8)
    for poly in polys:
        mask = cv2.fillPoly(mask, [poly], color=(255, 0, 0))
    mask_img = 0.9 * mask + img
    cv2.imwrite(save_name+'.png',mask_img)


def visualize(image, bboxes, category_ids, category_id_to_name, color=PRED_COLOR, show=False, save_name='result'):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color)
    cv2.imwrite(save_name+'.png',img[...,::-1])
    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    return img

def denorm(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std =  torch.tensor(std).view(-1, 1, 1)
    
    x = x * std + mean
    return x.permute(1, 2, 0)
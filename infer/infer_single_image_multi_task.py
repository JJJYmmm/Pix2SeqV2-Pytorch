import cv2
import os
import json
import pickle
import argparse
import torch
import numpy as np
import albumentations as A
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
sys.path.append('../')

from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, generate_with_prompt, postprocess, postprocess_caption, postprocess_segmentation, postprocess_keypoint
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize, visualize_mask, visualize_keypoint
from utils import seed_everything, adjust_box_transform2origin
from dataset.transforms import affine_points
from dataset.build_captioning_vocab import Vocabulary
from dataset.coco_object_detection import get_transform_test as object_detection_transform
from dataset.img_captioning import get_transform_valid as captioning_transform
from dataset.coco_segmentation import get_transform_valid as segmentation_transform
from dataset.coco_keypoint import get_transform_valid as keypoint_transform

parser = argparse.ArgumentParser("Infer single image with multi-task(detection, captioning, segmentation, keypoint)")
parser.add_argument("--image", type=str, help="Path to image", default="../images/person_segmentation.png")

if __name__ == '__main__':
    # fix seed
    seed_everything(42) 
    
    # prepare coco label
    assert os.path.exists(CFG.coco_label_path), "json file {} dose not exist.".format(CFG.coco_label_path)
    with open(CFG.coco_label_path, 'r') as f:
        id2cls = json.load(f) # num = 90,exclude background(id=0)
        id2cls = {int(i) : cls_name for i, cls_name in id2cls.items()}
    cls2id = {cls_name : i for i, cls_name in id2cls.items()}
    max_cls_id = max(id2cls.keys())

    # prepare vocab
    with open(CFG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)   

    # load model
    tokenizer = Tokenizer(num_classes=max_cls_id+1, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code
    # customed for image captioning
    tokenizer.vocab_size = 6000

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.multi_task_weight_path, map_location=CFG.device))
    print(msg)
    model.eval()

    # load data
    img_path = parser.parse_args().image
    ori_img = cv2.imread(img_path)[..., ::-1]

    ####################################
    # captioning
    ####################################
    transform_img = captioning_transform(CFG.img_size)(image=ori_img)['image'].unsqueeze(0)
    
    with torch.no_grad():
        print('Captioning:')
        for _ in range(3):
            batch_preds, batch_confs = generate(
                model, transform_img, tokenizer, max_len=CFG.max_len-1, top_k=10, top_p=0.8,begin_symbol=tokenizer.task_ids['captioning'])
            captions = postprocess_caption(
                batch_preds, tokenizer, vocab)
            print(captions)

    ####################################
    # object detection
    ####################################
    transform_img = object_detection_transform(CFG.img_size)(image=ori_img)['image'].unsqueeze(0)
    
    with torch.no_grad():
        batch_preds, batch_confs = generate(
            model, transform_img, tokenizer, max_len=CFG.max_len-1, top_k=10, top_p=0.8,begin_symbol=tokenizer.task_ids['detection'])
        bboxes, labels, confs = postprocess(
            batch_preds, batch_confs, tokenizer)

    # get first batch(single image) and adjust box from transforms to origin size
    bboxes = adjust_box_transform2origin(bboxes[0],ori_img.shape)
    labels = labels[0]

    # filter invalid bbox
    valid_mask = bboxes >= 0
    valid_mask = np.min(valid_mask,axis=1)
    bboxes = bboxes[valid_mask]
    labels = labels[valid_mask]

    visualize(ori_img, bboxes, labels, id2cls, show=False, save_name='obeject_detection')
    
    ####################################
    # prepare bboxes and labels(x_min,y_min,w,h)
    ####################################
    # xyxy to xywh
    bboxes[:, [2, 3]] -= bboxes[:, [0, 1]]

    person_mask = (labels == cls2id['person'])
    predict_cls = [id2cls[label] for label in labels.tolist()]

    print('Bounding boxes:')
    print(bboxes)
    print('Labels:')
    print(predict_cls)

    ####################################
    # segmentation
    ####################################
    seg_transforms = segmentation_transform(CFG.img_size)

    # get batch input
    bbox_imgs = []
    prompts = []
    trans_matrix = []
    for box in bboxes:
        target = dict(box=box,)
        transform_img, person_info = seg_transforms(ori_img, target)
        prompt = tokenizer.get_segmentation_prompt(person_info)
        bbox_imgs.append(transform_img)
        prompts.append(torch.LongTensor(prompt))
        trans_matrix.append(torch.tensor(person_info['reverse_trans']))

    if len(bbox_imgs) != 0:
        bbox_imgs = torch.stack(bbox_imgs, dim=0)
        prompts = torch.stack(prompts, dim=0)
        trans_matrix = torch.stack(trans_matrix, dim=0)
        # xywh to xyxy
        bboxes_temp = bboxes.copy()
        bboxes_temp[:, [2, 3]] += bboxes_temp[:, [0, 1]]
        bboxes_temp = torch.from_numpy(bboxes_temp)
        
        batch_data = TensorDataset(bbox_imgs, prompts,trans_matrix,bboxes_temp)

        batch_loader = DataLoader(dataset=batch_data, batch_size=CFG.batch_size, shuffle=False)

        all_polys = []
        with torch.no_grad():
            for _ in range(4): # follow the paper setting
                for imgs, prompts,trans_matrix,bboxes in batch_loader:
                    # max_len = 295 because segmentation need more steps to generate coords
                    batch_preds, _ = generate_with_prompt(
                        model, bbox_imgs, prompts, max_len=295, top_k=10, top_p=0.8, begin_symbol=tokenizer.task_ids['segmentation'])
                    segmentations = postprocess_segmentation(
                        batch_preds, tokenizer)
                    # segmentation[0] : list [poly_number(1),points,2]
                    for idx, batch in enumerate(segmentations):
                        transfomed_polys = []
                        bounding = np.array(bboxes[idx])
                        # print(bounding)
                        for poly in batch:
                            transformed_poly = affine_points(np.array(poly),trans_matrix[idx]).astype(np.int32)
                            transformed_poly[:,0] = np.fmin(np.fmax(transformed_poly[:,0],bounding[0]),bounding[2])
                            transformed_poly[:,1] = np.fmin(np.fmax(transformed_poly[:,1],bounding[1]),bounding[3])
                            transfomed_polys.append(transformed_poly)
                        # print(transfomed_polys)
                        all_polys.extend(transfomed_polys)

        visualize_mask(ori_img[...,::-1],all_polys,save_name='instance_segmentation')

    ####################################
    # keypoint
    ####################################
    keypoint_transforms = keypoint_transform(CFG.img_size)

    # get batch input
    bbox_imgs = []
    prompts = []
    trans_matrix = []
    for box in bboxes[person_mask]:
        target = dict(box=box,)
        transform_img, person_info = keypoint_transforms(ori_img, target)
        prompt = tokenizer.get_keypoint_prompt(person_info)
        bbox_imgs.append(transform_img)
        prompts.append(torch.LongTensor(prompt))
        trans_matrix.append(torch.tensor(person_info['reverse_trans']))
    
    if len(bbox_imgs) != 0: 
        bbox_imgs = torch.stack(bbox_imgs, dim=0)
        prompts = torch.stack(prompts, dim=0)
        trans_matrix = torch.stack(trans_matrix, dim=0)

        batch_data = TensorDataset(bbox_imgs, prompts,trans_matrix)

        batch_loader = DataLoader(dataset=batch_data, batch_size=CFG.batch_size, shuffle=False)

        keypoint_list = []
        with torch.no_grad():
            for _ in range(1):
                for imgs, prompts,trans_matrix in batch_loader:
                    # max_len = 295 because segmentation need more steps to generate coords
                    batch_preds, _ = generate_with_prompt(
                        model, bbox_imgs, prompts, max_len=CFG.generation_steps, top_k=10, top_p=0.8, begin_symbol=tokenizer.task_ids['keypoint'])
                    keypoints = postprocess_keypoint(
                        batch_preds, tokenizer)
                    print('Keypoint list:')
                    # [0,0] means invisible
                    print(keypoints)
                    for idx, batch in enumerate(keypoints):
                        if batch == None or len(batch) % 2 != 0:
                            continue
                        batch = np.array(batch).reshape(-1,2)
                        zero_mask = (batch == 0).min(axis=1) # get invisible token mask
                        affined_points = affine_points(batch, trans_matrix[idx])
                        affined_points[zero_mask] = 0 # recover invisible tokens
                        keypoint_list.append(affined_points)

        visualize_keypoint(ori_img[...,::-1],keypoint_list,save_name='keypoint_detection')


import cv2
import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import albumentations as A

import sys
sys.path.append('../')
from dataset.coco_object_detection import CoCoDetectionTest
from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, postprocess
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize
from utils import seed_everything, adjust_box_transform2origin
from dataset.coco_object_detection import get_transform_test as object_detection_transform

parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="../images/person.jpg")

if __name__ == '__main__':
    seed_everything(42) 
    assert os.path.exists(CFG.coco_label_path), "json file {} dose not exist.".format(CFG.coco_label_path)
    with open(CFG.coco_label_path, 'r') as f:
        id2cls = json.load(f) # num = 90,exclude background(id=0)
        id2cls = {int(i) : cls_name for i, cls_name in id2cls.items()}
    cls2id = {cls_name : i for i, cls_name in id2cls.items()}
    max_cls_id = max(id2cls.keys())
    print(max_cls_id+1)     

    tokenizer = Tokenizer(num_classes=max_cls_id+1, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.coco_weight_path, map_location=CFG.device))
    print(msg)
    model.eval()

    # load data
    img_path = parser.parse_args().image
    ori_img = cv2.imread(img_path)[..., ::-1]

    transform_img = object_detection_transform(CFG.img_size)(image=ori_img)['image'].unsqueeze(0)

    with torch.no_grad():
        batch_preds, batch_confs = generate(
            model, transform_img, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
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

    predict_cls = [id2cls[label] for label in labels.tolist()]

    # show predictions
    print('Bounding boxes:')
    print(bboxes)
    print('Labels:')
    print(predict_cls)

    visualize(ori_img, bboxes, labels, id2cls, show=False, save_name='obeject_detection')


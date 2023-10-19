import cv2
import argparse
import torch
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('../')
from dataset.voc_dataset import VOCDatasetTest
from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, postprocess
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize
from utils import seed_everything, adjust_box_transform2origin
from dataset.coco_object_detection import get_transform_test as object_detection_transform

parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="../images/desk.jpg")

if __name__ == '__main__':
    seed_everything(42) 
    with open(CFG.voc_label_path, 'r') as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    tokenizer = Tokenizer(num_classes=len(classes), num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    # if you use weight:voc_object_detection.pth, you should change the vocab size of tokenizer,from https://github.com/moein-shariatnia/Pix2Seq
    tokenizer.vocab_size = 384 + 20 + 3

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.voc_weight_path, map_location=CFG.device))
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

    # xyxy to xywh
    bboxes[:, [2, 3]] -= bboxes[:, [0, 1]]
    predict_cls = [id2cls[label] for label in labels.tolist()]

    # show predictions
    print('Bounding boxes:')
    print(bboxes)
    print('Labels:')
    print(predict_cls)

    visualize(ori_img, bboxes, labels, id2cls, show=False, save_name='obeject_detection')

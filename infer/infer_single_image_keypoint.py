import cv2
import torch
import numpy as np
import argparse

import sys,os
sys.path.append('../')
sys.path.append('./')
from utils import seed_everything
from test_utils import generate_with_prompt, postprocess_keypoint
from dataset.coco_keypoint import CocoKeypoint,get_transform_valid 
from dataset.transforms import affine_points
from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from config import CFG
from visualize import visualize_keypoint, denorm


parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="../images/person_keypoint.jpg")
parser.add_argument('--box',type=list,help="person bbox",default=[148.92517,304.1253,182.14099,198.85117]) # [159.19, 309.5, 165.16, 188.9]

if __name__ == '__main__':
    seed_everything(42)

    tokenizer = Tokenizer(num_classes=CFG.num_classes, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.coco_keypoint_weight_path, map_location=CFG.device))
    print(msg)
    model.eval()

    img_path = parser.parse_args().image
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    
    box = parser.parse_args().box
    target = dict(box=box)

    transforms = get_transform_valid(CFG.img_size)
    img, person_info = transforms(ori_img,target)

    prompt = tokenizer.get_keypoint_prompt(person_info)

    # just for batch input
    img = img.unsqueeze(0) 
    prompt = torch.LongTensor(prompt).unsqueeze(0)
    
    with torch.no_grad():
        batch_preds, batch_confs = generate_with_prompt(
            model, img, prompt, max_len=CFG.generation_steps, top_k=0, top_p=1)
        keypoint_list = postprocess_keypoint(
            batch_preds, tokenizer)
    
    keypoints = np.array(keypoint_list[0]).reshape(-1,2)

    # view transformed picture
    # tensor -> numpy.array -> denorm -> float2int ->rgb2gbr
    # img = denorm(img.squeeze(0)).numpy()[...,::-1]
    # img *= 255
    # img = img.astype(np.uint8)
    # visualize_keypoint(img, keypoints) 
    
    # view box and corresponding keypoints 
    box = [int(item) for item in box]
    cv2.rectangle(ori_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]),color=(255,0,0), thickness=1)
    visualize_keypoint(ori_img[...,::-1],[affine_points(keypoints,person_info['reverse_trans'])],'keypoint_detection')

    

import cv2
import torch
import numpy as np
import argparse

import sys,os
sys.path.append('../')
sys.path.append('./')
from utils import seed_everything
from test_utils import generate_with_prompt, postprocess_segmentation
from dataset.coco_segmentation import get_transform_valid 
from dataset.transforms import affine_points
from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from config import CFG
from visualize import visualize_mask, denorm


parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="../images/person_segmentation.png")
parser.add_argument('--box',type=list,help="bbox",default=[260.67883,97.12359,96.91907,310.8094])

if __name__ == '__main__':
    seed_everything(42)
    CFG.generation_steps = 295 # segmentation should have a longer sequence

    tokenizer = Tokenizer(num_classes=CFG.num_classes, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.coco_seg_weight_path, map_location=CFG.device))
    print(msg)
    model.eval()

    img_path = parser.parse_args().image
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    
    box = parser.parse_args().box
    target = dict(box=box,)

    transforms = get_transform_valid(CFG.img_size)
    img, person_info = transforms(ori_img,target)

    prompt = tokenizer.get_segmentation_prompt(person_info)

    # just for batch input
    img = img.unsqueeze(0) 
    prompt = torch.LongTensor(prompt).unsqueeze(0)
    
    all_polys = [] 
    with torch.no_grad():
        for _ in range(8): # follow the paper
            batch_preds, batch_confs = generate_with_prompt(
                model, img, prompt, max_len=CFG.generation_steps, top_k=10, top_p=0.8)
            segmentations = postprocess_segmentation(
                batch_preds, tokenizer)
            # segmentation[0] : list [poly_number(1),points,2]
            # print(len(segmentations[0][0]))
            all_polys.extend(segmentations[0])

    box = [int(item) for item in box]
    cv2.rectangle(ori_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]),color=(255,0,0), thickness=1)

    # reverse affine
    transformed_poly = []
    for idx, poly in enumerate(all_polys):
        transformed_poly.append(affine_points(np.array(poly),person_info['reverse_trans']).astype(np.int32))

    visualize_mask(ori_img[...,::-1].copy(),transformed_poly,'instance_segmentation')



    

import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io

import sys
sys.path.append('../')
from dataset.coco_object_detection import CocoDetection,get_transform_valid
from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, postprocess, collate_fn
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize

parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="./images/cars.jpg")

if __name__ == '__main__':

    CFG.generation_steps = 196 # generate more

    if not os.path.exists('./result_coco.json'):
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

        valid_ds = CocoDetection(root=CFG.dir_root,dataset='val',transforms=get_transform_valid(CFG.img_size), tokenizer=None)

        validloader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG.batch_size,
            collate_fn = collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            res = []
            for input, target in tqdm(validloader): # per batch 
                batch_preds, batch_confs = generate(
                    model, input, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
                bboxes, labels, confs = postprocess(
                    batch_preds, batch_confs, tokenizer)
                for bs, boxes in enumerate(bboxes): # per image
                    if boxes is None:
                        continue
                    if len(boxes) > 0:

                        # correct box by origin size
                        origin_h, origin_w = target[bs]['origin_size']
                        scale_fct = np.array([1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size,1.0*origin_w/CFG.img_size, 1.0*origin_h/CFG.img_size],dtype='float32')

                        if origin_w > origin_h:
                            pad_top = int((CFG.img_size - origin_h/scale_fct[0])/2.0)
                            boxes[:,[1, 3]] -= pad_top
                            boxes *= scale_fct[0]
                        else:
                            pad_left = int((CFG.img_size - origin_w/scale_fct[1])/2.0)
                            boxes[:,[0,2]] -= pad_left
                            boxes *= scale_fct[1]
                        
                        # xyxy to xywh
                        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
                        for k,box in enumerate(boxes): # per box
                            temp = target[bs]
                            temp['image_id'] = int(temp['image_id'])
                            temp['category_id'] = int(labels[bs][k])
                            temp['bbox'] = box.tolist()
                            temp['score'] = float(confs[bs][k])
                            res.append(temp)
                            
            with open('./result_coco.json','w') as f:
                json.dump(res,f)

    cocoGt = COCO('/mnt/MSCOCO/annotations/instances_val2017.json')
    cocoDt = cocoGt.loadRes('./result_coco.json')  
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

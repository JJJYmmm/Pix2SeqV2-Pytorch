import os
import json
import torch
import argparse
from torch import nn
import numpy as np
from glob import glob
from transformers import get_linear_schedule_with_warmup
import pickle

import sys
sys.path.append('../')
sys.path.append('./')
from utils import seed_everything

from dataset.coco_object_detection import get_loaders as detection_loaders
from dataset.img_captioning import get_loaders as img_caption_loaders
from dataset.coco_keypoint import get_loaders as keypoint_loaders
from dataset.coco_segmentation import get_loaders as segmentation_loaders

from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from multi_task import get_multi_task_loaders, get_multi_task_weights, train_eval_multi_task
from config import CFG
from dataset.build_captioning_vocab import Vocabulary

parser = argparse.ArgumentParser("train multi-tasks")
parser.add_argument("--task", type = lambda s: [item for item in s.split(',')], help="selected tasks,[detection, keypoint, segmentation, captioning]", default="detection,keypoint,segmentation,captioning")

if __name__ == '__main__':
    seed_everything(42)

    tokenizer = Tokenizer(num_classes=CFG.num_classes, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    # customed for image captioning
    tokenizer.vocab_size = 6000
    
    train_loaders,valid_loaders = get_multi_task_loaders(tokenizer, parser.parse_args().task)    
   
    task_weights = get_multi_task_weights(parser.parse_args().task) 

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(
        [{'params':model.parameters(),'initial_lr':CFG.lr}], lr=CFG.lr, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * \
        (max([len(loader.dataset) for loader in train_loaders.values()]) // CFG.batch_size)

    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps,last_epoch=CFG.start_epoch-1)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [i for i in range(CFG.epochs)], 0.2, last_epoch=CFG.start_epoch)

    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)
    criterion = criterion.to(CFG.device)

    # torch.autograd.set_detect_anomaly(True)
    train_eval_multi_task(model,
               train_loaders,
               valid_loaders,
               tokenizer.task_ids,
               task_weights,
               criterion,
               optimizer,
               lr_scheduler=lr_scheduler,
               step = 'batch',
               dataset='multi-task')
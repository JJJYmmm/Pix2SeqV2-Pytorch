import json
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
import pickle

import sys,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append('../')
sys.path.append('./')
from utils import seed_everything
from dataset.coco_segmentation import get_loaders
from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from engine import train_eval
from config import CFG



if __name__ == '__main__':
    seed_everything(42)

    with open(CFG.keypoints_path) as f:
        person_kps_info = json.load(f)

    tokenizer = Tokenizer(num_classes=CFG.num_classes, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    train_loader, valid_loader = get_loaders(
       CFG.dir_root, tokenizer,person_kps_info, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
    
    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(
        [{'params':model.parameters(),'initial_lr':CFG.lr}], lr=CFG.lr, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * \
        (len(train_loader.dataset) // CFG.batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps,last_epoch=CFG.start_epoch-1)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [i for i in range(CFG.epochs)], 0.2, last_epoch=CFG.start_epoch)

    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)
    criterion = criterion.to(CFG.device)

    # torch.autograd.set_detect_anomaly(True)
    train_eval(model,
               train_loader,
               valid_loader,
               criterion,
               optimizer,
               lr_scheduler=lr_scheduler,
               step = 'batch',
               dataset='segmentation')
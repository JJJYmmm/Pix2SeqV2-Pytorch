import cv2
import argparse
import torch
import pickle
import albumentations as A

import sys
sys.path.append('../')
sys.path.append('./')
from dataset.img_captioning import CoCoCaptioningTest,get_loaders
from model import Encoder, Decoder, EncoderDecoder
from test_utils import generate, postprocess_caption
from tokenizer import Tokenizer
from dataset.build_captioning_vocab import Vocabulary
from config import CFG
from utils import seed_everything


parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="../images/car.jpg")

if __name__ == '__main__':
    seed_everything(42) 

    # Load vocabulary wrapper
    with open(CFG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)   

    # customed for image caption
    CFG.max_len = 100

    tokenizer = Tokenizer(num_classes=CFG.num_classes, num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code
    
    # customed for image caption
    tokenizer.vocab_size = 6000

    img_paths = [parser.parse_args().image]
    test_dataset = CoCoCaptioningTest(img_paths, size=CFG.img_size)

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        CFG.coco_caption_weight_path, map_location=CFG.device))
    print(msg)
    model.eval()

    x = test_dataset[0].unsqueeze(dim=0)

    with torch.no_grad():
        for i in range(3):
            batch_preds, batch_confs = generate(
                model, x, tokenizer, max_len=CFG.max_len-1, top_k=10, top_p=0.8)
            captions = postprocess_caption(
                batch_preds, tokenizer, vocab)
            print(captions)


    # train_loader, valid_loader = get_loaders(
    #    CFG.dir_root, tokenizer,vocab, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
    
    # with torch.no_grad():
    #     sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
    #     for x, y in valid_loader:
    #         x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)

    #         y_input = y[:, :-1]
    #         y_expected = y[:, 1:]

    #         preds = model(x, y_input)
    #         print(list(map(lambda x:vocab.get_word(x-550),sample(preds[0]).squeeze(1).tolist())))
    #         print(list(map(lambda x:vocab.get_word(x-550),y_expected[0].tolist())))
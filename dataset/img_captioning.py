import cv2
import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import albumentations as A
from functools import partial
from dataset.build_captioning_vocab import Vocabulary
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from albumentations.pytorch.transforms import ToTensorV2
                       
def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size,size),
        A.ToFloat(max_value=255),
        ToTensorV2()
    ])

def get_transform_valid(size):
    return A.Compose([
            A.RandomResizedCrop(size,size),
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])
    
class CoCoCaptioning(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, dataset="train",transforms=None,tokenizer=None, vocab=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        super(CoCoCaptioning,self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = "captions_{}2017.json".format(dataset)
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, "{}2017".format(dataset))
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.coco = COCO(self.anno_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_root, path)).convert('RGB')
        image = np.asarray(image)
        if self.transforms is not None:
            transformd = self.transforms(image=image)
            image = transformd['image']

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        # todo : check caption's max len
        caption.append(self.tokenizer.BOS_code)
        caption.extend([vocab(token) + self.tokenizer.text_id_shift for token in tokens])
        caption.append(self.tokenizer.EOS_code)

        # caption2 = caption.copy()
        # caption2 = list(map(lambda x:vocab.get_word(x - self.tokenizer.text_id_shift),caption2))
        # print(caption2)

        target = torch.LongTensor(caption)

        # init_len = 0, object detection and captioning have no prompts
        return image, target, 0

    def __len__(self):
        return len(self.ids)

def collate_fn(batch,max_len,pad_idx):
    image_batch, seq_batch, init_len_batch = [], [], []
    for image, seq, len in batch:
        image_batch.append(image)
        seq_batch.append(seq)
        init_len_batch.append(len)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                        seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch, init_len_batch
   

def get_loaders(dir_root, tokenizer, vocab, img_size, batch_size, max_len, pad_idx, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    train_ds = CoCoCaptioning(root=dir_root,dataset='train',transforms=get_transform_train(img_size), tokenizer=tokenizer,vocab=vocab)
    
    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = CoCoCaptioning(root=dir_root,dataset='val',transforms=get_transform_valid(img_size), tokenizer=tokenizer,vocab=vocab)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader

class CoCoCaptioningTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        super(CoCoCaptioningTest, self).__init__()
        self.img_paths = img_paths
        self.transforms = A.Compose([
                A.RandomResizedCrop(size,size),
                A.ToFloat(max_value=255),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)[..., ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img

    def __len__(self):
        return len(self.img_paths)
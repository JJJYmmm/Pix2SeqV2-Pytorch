import cv2
from functools import partial

import torch
import albumentations as A
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence
from albumentations.pytorch import ToTensorV2

def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(size, size),
        # A.LongestMaxSize(max_size=size, interpolation=1),
        # A.ToFloat(max_value=255),
        A.Normalize(),
        # A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_transform_valid(size):
    return A.Compose([
        A.Resize(size, size),
        # A.LongestMaxSize(max_size=size, interpolation=1),
        # A.ToFloat(max_value=255),
        A.Normalize(),
        # A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None, tokenizer=None):
        self.ids = df['id'].unique()
        self.df = df
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = self.df[self.df['id'] == self.ids[idx]]
        img_path = sample['img_path'].values[0]

        img = cv2.imread(img_path)[..., ::-1]
        labels = sample['label'].values
        bboxes = sample[['xmin', 'ymin', 'xmax', 'ymax']].values

        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # img = torch.FloatTensor(img).permute(2, 0, 1) # replaced by ToTensorV2

        if self.tokenizer is not None:
            seqs = self.tokenizer.encode_box(labels, bboxes)
            seqs = torch.LongTensor(seqs)
            # init_len = 0, object detection and captioning have no prompts
            return img, seqs, 0

        return img, labels, bboxes

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

def split_df(df, n_folds=5, training_fold=0):
    mapping = df.groupby("id")['img_path'].agg(len).to_dict()
    df['stratify'] = df['id'].map(mapping)

    kfold = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=42)

    for i, (_, val_idx) in enumerate(kfold.split(df, y=df['stratify'], groups=df['id'])):
        df.loc[val_idx, 'fold'] = i

    train_df = df[df['fold'] != training_fold].reset_index(drop=True)
    valid_df = df[df['fold'] == training_fold].reset_index(drop=True)

    return train_df, valid_df


def get_loaders(train_df, valid_df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2):

    train_ds = VOCDataset(train_df, transforms=get_transform_train(
        img_size), tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = VOCDataset(valid_df, transforms=get_transform_valid(
        img_size), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader


class VOCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_paths, size):
        self.img_paths = img_paths
        self.transforms = A.Compose([
            A.Resize(size, size),
            # A.ToFloat(max_value=255),
            A.Normalize(),
            ToTensorV2()
            ])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)[..., ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        # img = torch.FloatTensor(img).permute(2, 0, 1) # replaced by ToTensorV2

        return img

    def __len__(self):
        return len(self.img_paths)

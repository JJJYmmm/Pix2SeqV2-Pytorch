import torch
import pickle
from tqdm import tqdm
import json
from tensorboardX import SummaryWriter

from utils import AvgMeter, get_lr
from config import CFG
from dataset.coco_object_detection import get_loaders as detection_loaders
from dataset.img_captioning import get_loaders as img_caption_loaders
from dataset.coco_keypoint import get_loaders as keypoint_loaders
from dataset.coco_segmentation import get_loaders as segmentation_loaders

def get_multi_task_weights(tasks):

    assert set(tasks) <= set(['detection', 'keypoint', 'segmentation', 'captioning'])

    task_weights = {
        'detection': 0.1782,
        'segmentation': 0.7128,
        'captioning': 0.0099,
        'keypoint': 0.01
    }

    selected_tasks = dict([(task, task_weights.get(task, None)) for task in tasks])

    total = sum([weight for weight in selected_tasks.values()])
    for k in selected_tasks.keys():
        selected_tasks[k] /= total
    
    return selected_tasks
        
    

def get_multi_task_loaders(tokenizer,tasks):

    assert set(tasks) <= set(['detection', 'keypoint', 'segmentation', 'captioning'])

    # Load vocabulary wrapper
    with open(CFG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)   

    with open(CFG.keypoints_path) as f:
        person_kps_info = json.load(f)

    train_loaders = {}
    valid_loaders = {}

    if 'detection' in tasks:
        detection_train_loader, detection_valid_loader = detection_loaders(
        CFG.dir_root, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
        train_loaders['detection'] = detection_train_loader
        valid_loaders['detection'] = detection_valid_loader
   
    if 'keypoint' in tasks: 
        keypoint_train_loader, keypoint_valid_loader = keypoint_loaders(
        CFG.dir_root, tokenizer,person_kps_info, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
        train_loaders['keypoint'] = keypoint_train_loader
        valid_loaders['keypoint'] = keypoint_valid_loader

    if 'segmentation' in tasks: 
        segmentation_train_loader, segmentation_valid_loader = segmentation_loaders(
        CFG.dir_root, tokenizer,person_kps_info, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
        train_loaders['segmentation'] = segmentation_train_loader
        valid_loaders['segmentation'] = segmentation_valid_loader

    if 'captioning' in tasks:
        img_caption_train_loader, img_caption_valid_loader = img_caption_loaders(
        CFG.dir_root, tokenizer,vocab, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)
        train_loaders['captioning'] = img_caption_train_loader
        valid_loaders['captioning'] = img_caption_valid_loader
    
    return train_loaders, valid_loaders

def train_eval_multi_task(model, 
               train_loaders,
               valid_loaders,
               task_ids,
               task_weights,
               criterion, 
               optimizer, 
               lr_scheduler,
               step,
               dataset):
    
    best_loss = float(CFG.resume_loss)

    if CFG.resume:
        msg = model.load_state_dict(torch.load(CFG.resume_path, map_location=CFG.device))
        print(msg)

    writer = SummaryWriter(log_dir='./logs',flush_secs=60)
    
    for epoch in range(CFG.start_epoch,CFG.epochs):
        print(f"Epoch {epoch + 1}")
        
        train_loss = train_epoch_multi_task(model, train_loaders, task_ids, task_weights, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, epoch=epoch, logger=writer)

        torch.save(model.state_dict(), f'train_loss_{dataset}_epoch{epoch+1}_loss{train_loss:.4}.pth')

        valid_loss = valid_epoch_multi_task(model, valid_loaders, task_ids, task_weights, criterion)
        print(f"Valid loss: {valid_loss:.3f}")

        writer.add_scalar('Validation loss',valid_loss,epoch)
        
        if step == 'epoch':
            pass
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'best_valid_loss_{dataset}_epoch{epoch+1}_loss{best_loss:.4}.pth')
            print("Saved Best Model")


def train_epoch_multi_task(model, train_loaders:dict, task_ids, task_weights, optimizer, lr_scheduler, criterion, epoch, logger=None):
    model.train()
    loss_meter = AvgMeter()

    # get longest data loader
    epoch_size = 0
    longest_loader = None
    for name, loader in train_loaders.items():
        if len(loader) > epoch_size:
            epoch_size = len(loader)
            longest_loader = name
    
    loader_iters = dict()
    for k, v in train_loaders.items():
        if k != longest_loader:
            loader_iters[k] = iter(v)

    tqdm_object = tqdm(train_loaders[longest_loader], total=len(train_loaders[longest_loader]))
 
    for iteration,(x, y, init_lens) in enumerate(tqdm_object):

        optimizer.zero_grad()
            
        total_loss = torch.zeros(1, requires_grad=False, device=CFG.device)
        total_batch = x.size(0)
        
        loss = cal_loss_multi_task(model, criterion, x, y, init_lens, task_id = task_ids[longest_loader])
        total_loss = total_loss + loss.item() * task_weights[longest_loader]
        loss *= task_weights[longest_loader]
        loss.backward()

        for k, v in loader_iters.items():
            try:
                (x, y, init_lens) = next(v)
            except StopIteration:
                loader_iters[k] = iter(train_loaders[k])
                (x, y, init_lens) = next(loader_iters[k])
            total_batch += x.size(0)
            loss = cal_loss_multi_task(model, criterion, x, y, init_lens, task_id=task_ids[k])
            total_loss = total_loss + loss.item() * task_weights[k]
            loss *= task_weights[k]
            loss.backward()

        # total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_meter.update(total_loss.item(), total_batch)
        
        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")
        
        if logger is not None:
            logger.add_scalar('Train_loss', total_loss, (epoch*epoch_size + iteration))
    
    return loss_meter.avg

def valid_epoch_multi_task(model, valid_loaders, task_ids, task_weights, criterion):
    model.eval()
    loss_meter = AvgMeter()
    # get longest data loader
    epoch_size = 0
    longest_loader = None
    for name, loader in valid_loaders.items():
        if len(loader) > epoch_size:
            epoch_size = len(loader)
            longest_loader = name
    
    loader_iters = dict()
    for k, v in valid_loaders.items():
        if k != longest_loader:
            loader_iters[k] = iter(v)

    tqdm_object = tqdm(valid_loaders[longest_loader], total=len(valid_loaders[longest_loader]))

    with torch.no_grad():
        for x, y, init_lens in tqdm_object:
            
            total_loss = torch.zeros(1, requires_grad=False, device=CFG.device)
            total_batch = x.size(0)
            
            loss = cal_loss_multi_task(model, criterion, x, y, init_lens, task_id = task_ids[longest_loader])
            total_loss = total_loss + loss.item() * task_weights[longest_loader]

            for k, v in loader_iters.items():
                try:
                    (x, y, init_lens) = next(v)
                except StopIteration:
                    continue # different from those used during training

                total_batch += x.size(0)
                loss = cal_loss_multi_task(model, criterion, x, y, init_lens, task_id=task_ids[k])
                total_loss = total_loss + loss.item() * task_weights[k]

            loss_meter.update(total_loss.item(), total_batch)
    
    return loss_meter.avg


def cal_loss_multi_task(model, criterion, x, y, init_lens, task_id):
    x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)

    # no use
    # add task prompt, and cut the last element y[:,1:-1](because length of y shall be CFG.max_len)
    # task_prompt = torch.full((y.size(0), 1), task_id).to(CFG.device, non_blocking=True)
    # y = torch.concat([y[:,0].unsqueeze(1), task_prompt, y[:,1:-1]], dim=1)

    # replace BOS with task prompt
    task_prompt = torch.full((y.size(0), 1), task_id).to(CFG.device, non_blocking=True)
    y = torch.concat([task_prompt, y[:,1:]], dim=1)

    y_input = y[:, :-1]
    y_expected = y[:, 1:]
    
    preds = model(x, y_input)
    
    # jump init tokens
    mask = torch.ones((x.size(0),y.size(1)-1),dtype=torch.uint8)
    for id, init_len in enumerate(init_lens):
        # + 1 : add task prompt
        mask[id,:init_len + 1] = 0
    mask = mask.bool().to(preds.device) 

    # loss = criterion(preds[mask].reshape(-1, preds.shape[-1]), y_expected[mask].reshape(-1))
    loss = criterion(preds[mask], y_expected[mask])
    
    return loss

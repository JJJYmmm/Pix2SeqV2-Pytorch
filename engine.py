from tqdm import tqdm
import torch

from utils import AvgMeter, get_lr
from config import CFG
from tensorboardX import SummaryWriter

def train_epoch(model, train_loader, optimizer, lr_scheduler, criterion, epoch, logger=None):
    model.train()
    loss_meter = AvgMeter()
    epoch_size = len(train_loader)
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for iteration,(x, y, init_lens) in enumerate(tqdm_object):
        x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
        # print(f'x:{x}')
        # print(f'y:{y}')

        # print(y[:,1]) 
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        
        preds = model(x, y_input)
        
        # if torch.any(torch.isnan(preds)):
            # print(x.shape)
            # print(y.shape)
            # print(preds)
            # exit(1)

        # jump init tokens
        mask = torch.ones((x.size(0),y.size(1)-1),dtype=torch.uint8)
        for id, init_len in enumerate(init_lens):
            mask[id,:init_len] = 0
        mask = mask.bool().to(preds.device) 

        # loss = criterion(preds[mask].reshape(-1, preds.shape[-1]), y_expected[mask].reshape(-1))
        loss = criterion(preds[mask], y_expected[mask])

        # print(f'preds:{preds}')
        # print(f'y_expect:{y_expected}')
        # print(f'loss:{loss}')
        
        optimizer.zero_grad()
        
        # with torch.autograd.detect_anomaly():
        loss.backward()

        # gradient clip 
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        loss_meter.update(loss.item(), x.size(0))
        
        lr = get_lr(optimizer)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=f"{lr:.5f}")

        if logger is not None:
            logger.add_scalar('Train_loss', loss, (epoch*epoch_size + iteration))
    
    return loss_meter.avg

def valid_epoch(model, valid_loader, criterion):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    with torch.no_grad():
        for x, y,init_lens in tqdm_object:
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)

            y_input = y[:, :-1]
            y_expected = y[:, 1:]

            preds = model(x, y_input)
            # jump init tokens
            mask = torch.ones((x.size(0),y.size(1)-1),dtype=torch.uint8)
            for id, init_len in enumerate(init_lens):
                mask[id,:init_len] = 0
            mask = mask.bool().to(CFG.device) 

            loss = criterion(preds[mask], y_expected[mask])


            loss_meter.update(loss.item(), x.size(0))
    
    return loss_meter.avg


def train_eval(model, 
               train_loader,
               valid_loader,
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
        
        train_loss = train_epoch(model, train_loader, optimizer, 
                                 lr_scheduler if step == 'batch' else None, 
                                 criterion, epoch=epoch, logger=writer)
        
        valid_loss = valid_epoch(model, valid_loader, criterion)
        print(f"Valid loss: {valid_loss:.3f}")

        writer.add_scalar('Validation loss',valid_loss,epoch)
        
        if step == 'epoch':
            pass
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'best_valid_loss_{dataset}_epoch{epoch+1}_loss{best_loss:.4}.pth')
            print("Saved Best Model")


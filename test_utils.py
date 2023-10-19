import torch
from transformers import top_k_top_p_filtering
from utils import box_cxcywh_to_xyxy
from config import CFG

def collate_fn(batch):
    img, target = tuple(zip(*batch))
    img = torch.stack(img)
    return (img,target)

def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1, begin_symbol=None):
    x = x.to(CFG.device)
    if begin_symbol == None:
        begin_symbol = tokenizer.BOS_code
    batch_preds = torch.ones(x.size(0), 1).fill_(begin_symbol).long().to(CFG.device)
    confs = []
    
    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
        
    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            ## If top_k and top_p are set to default, the following line does nothing!
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            # just for object detection
            if i % 4 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)
    # you can ignore confs when the task is not object detection 
    return batch_preds.cpu(), confs

def generate_with_prompt(model, x, batch_preds, max_len=50, top_k=0, top_p=1,begin_symbol=None):
    x = x.to(CFG.device)
    if begin_symbol != None:
        batch_preds[:,0] = begin_symbol
    batch_preds = batch_preds.to(CFG.device)


    confs = []
    
    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
        
    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            ## If top_k and top_p are set to default, the following line does nothing!
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            # just for object detection
            if i % 4 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)
    # you can ignore confs when the task is not object detection 
    return batch_preds.cpu(), confs

def postprocess(batch_preds, batch_confs, tokenizer):
    batch_preds[:,-1] = tokenizer.EOS_code # ensure a complete sentence
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_bboxes = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0 or EOS_idx ==1: # fix : invalid idx which EOS_idx = 0 or the model detect nothing which EOS_idx = 1
            all_bboxes.append(None)
            all_labels.append(None)
            all_confs.append(None)
            continue
        labels, bboxes = tokenizer.decode_box(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]
        
        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_confs.append(confs)
        
    return all_bboxes, all_labels, all_confs

def postprocess_caption(batch_preds, tokenizer, vocab):
    batch_preds[:,-1] = tokenizer.EOS_code # ensure a complete sentence
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    captions = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0 or EOS_idx ==1: # fix : invalid idx which EOS_idx = 0 or the model detect nothing which EOS_idx = 1
            captions.append(None)
            continue
        caption = []
        for word in batch_preds[i][1:EOS_idx]:
            caption.append(vocab.get_word(word.item()-tokenizer.text_id_shift))
        captions.append(caption)
        
    return captions

def postprocess_keypoint(batch_preds, tokenizer):
    batch_preds[:,-1] = tokenizer.EOS_code # ensure a complete sentence
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check, a person has 17 keypoints, a point has 2 coord, 4 box prompt
    invalid_idxs = ((EOS_idxs - 1) != 38).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
 
    keypoints = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0 or EOS_idx ==1: # fix : invalid idx which EOS_idx = 0 or the model detect nothing which EOS_idx = 1
            keypoints.append(None)
            continue
        keypoints_per_batch = tokenizer.decode_keypoint(batch_preds[i][:EOS_idx+1])
        keypoints.append(keypoints_per_batch)
        
    return keypoints

def postprocess_segmentation(batch_preds, tokenizer):
    batch_preds[:,-1] = tokenizer.EOS_code # ensure a complete sentence
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
 
    segmentations = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0 or EOS_idx ==1: # fix : invalid idx which EOS_idx = 0 or the model detect nothing which EOS_idx = 1
            segmentations.append(None)
            continue
        segmentation_per_batch = tokenizer.decode_segmentation(batch_preds[i][:EOS_idx+1])
        segmentations.append(segmentation_per_batch)
        
    return segmentations


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)



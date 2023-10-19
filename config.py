import torch

class CFG:

    # resume training
    resume = False
    resume_loss = float('inf')
    resume_path = ''
    start_epoch = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # model config 
    max_len = 300
    img_size = 384
    num_bins = img_size
    num_classes = 91
    
    batch_size = 16
    epochs = 2
    
    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
 
    # voc dataset params
    img_path = '../download/VOCdevkit/VOC2012/JPEGImages'
    xml_path = '../download/VOCdevkit/VOC2012/Annotations'
    voc_label_path = '../train/voc_classes.txt'
    voc_weight_path = '../weights/voc_object_detection.pth'

    # coco dataset params
    dir_root = '/mnt/MSCOCO'
    coco_label_path = '../train/coco91_indices.json'
    coco_weight_path = '../weights/coco_ob_wo_pixnorm.pth'

    # image captioning params
    vocab_path = '../train/vocab.pkl'
    coco_caption_weight_path = '../weights/coco_image_caption.pth'

    # keypoint detection params
    keypoints_path = '../train/person_keypoints.json'
    coco_keypoint_weight_path = '../weights/coco_keypoint.pth'
    num_joints = 17 # for coco dataset, no use

    # segmentation params
    coco_seg_weight_path = '../weights/coco_segmentation.pth'
    
    # multi task params
    multi_task_weight_path = '../weights/coco_multi_task.pth'
    
    
    # optim
    lr = 1e-4
    weight_decay = 1e-4
    
    # eval  
    generation_steps = 101
    run_eval = True
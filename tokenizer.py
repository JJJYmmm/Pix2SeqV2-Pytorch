import numpy as np
import torch

class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=500):

        # for coco
        # [0, 384) for bins
        # [384, 91+384) for classes
        # 475 for BOS
        # 476 for EOS
        # 477 for PAD
        # [478,500) for task id
        # [500,550) reserved:
        # # 500 for invisible coord in keypoints
        # # 501 for seperate token betweens polys
        # [550,...) for text id 

        self.num_classes = num_classes
        self.num_bins = num_bins

        self.width = width
        self.height = height

        self.max_len = max_len
        self.max_len_obj = int((self.max_len - 2) / 5) # max_len means the max len of seq, -2 means excluding bos and eos
        self.max_num_point = 128 # max point number for segmentation

        # solver negative token
        self.negative_solver = np.frompyfunc(lambda x: 0 if x < 0 else x, 1, 1) 

        self.BOS_code = num_classes + num_bins
        self.EOS_code = self.BOS_code + 1
        self.PAD_code = self.EOS_code + 1

        # train multi task 
        self.task_id_shift = self.PAD_code + 1
        self.task_ids = {
        'detection': self.task_id_shift,
        'segmentation': self.task_id_shift + 1,
        'captioning': self.task_id_shift + 2,
        'keypoint': self.task_id_shift + 3
        }

        self.text_id_shift = 550

        self.invisible_token = 500
        self.seperate_token = 501

        self.vocab_size = 2000 # big enough for all excluding text, in image captioning vocab_size = 6000

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')
    
    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1)

    def encode_box(self, labels: list, bboxes: list, shuffle=True):
        assert len(bboxes), "seq should not be empty(friendly to CrossEntropy Loss)"
        assert len(labels) == len(bboxes), "labels and bboxes must have the same length"
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        labels += self.num_bins
        labels = labels.astype('int')[:self.max_len_obj]

        bboxes[:, 0] = bboxes[:, 0] / self.width
        bboxes[:, 2] = bboxes[:, 2] / self.width
        bboxes[:, 1] = bboxes[:, 1] / self.height
        bboxes[:, 3] = bboxes[:, 3] / self.height

        bboxes = self.quantize(bboxes)[:self.max_len_obj]

        if shuffle:
            rand_idxs = np.arange(0, len(bboxes))
            np.random.shuffle(rand_idxs)
            labels = labels[rand_idxs]
            bboxes = bboxes[rand_idxs]

        tokenized = [self.BOS_code]
        for label, bbox in zip(labels, bboxes):
            tokens = list(bbox)
            tokens.append(label)

            tokenized.extend(list(map(int, tokens)))
        tokenized.append(self.EOS_code)

        init_len = 0
    
        return tokenized
    
    def decode_box(self, tokens: torch.tensor):
        """
        toekns: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        assert len(tokens) % 5 == 0, "invalid tokens"

        labels = []
        bboxes = []
        for i in range(4, len(tokens)+1, 5):
            label = tokens[i]
            bbox = tokens[i-4: i]
            labels.append(int(label))
            bboxes.append([int(item) for item in bbox])
        labels = np.array(labels) - self.num_bins
        bboxes = np.array(bboxes)
        bboxes = self.dequantize(bboxes)
        
        bboxes[:, 0] = bboxes[:, 0] * self.width
        bboxes[:, 2] = bboxes[:, 2] * self.width
        bboxes[:, 1] = bboxes[:, 1] * self.height
        bboxes[:, 3] = bboxes[:, 3] * self.height
        
        return labels, bboxes
    
    def encode_keypoint(self, person_info: dict):
        # init seq with bbox
        box = np.array(person_info['box'],dtype=np.float64)
        box[0] = min(max(0,box[0]),self.width)/self.width
        box[1] = min(max(0,box[1]),self.height)/self.height
        box[2] = min(max(0,box[2]),self.width)/self.width
        box[3] = min(max(0,box[3]),self.height)/self.height
        box = self.quantize(box)

        # keypoints
        keypoints = person_info['keypoints']
        vis = person_info['visible']
        keypoint_list = []
        invisible = [self.invisible_token, self.invisible_token]
        for state,keypoint in zip(vis,keypoints):
            if keypoint[0] < 0 or keypoint[1] < 0 or keypoint[0] > self.width or keypoint[1] > self.height:
                keypoint_list.extend(invisible)
            elif state > 0.5:
                keypoint_list.extend(keypoint)
            else:
                keypoint_list.extend(invisible)

        seq = [self.BOS_code]
        seq.extend(list(map(int,box)))
        seq.extend(list(map(int, keypoint_list)))
        seq.append(self.EOS_code)
        init_len = len(box)        
        return seq, init_len
    
    def decode_keypoint(self, tokens):
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[5:-1] # jump bos and box prompt
        assert len(tokens) == 34 , "invalid tokens"
        keypoint_list = []
        for i in range(0,len(tokens),2):
            x = tokens[i]
            y = tokens[i+1]
            if x == self.invisible_token or y == self.invisible_token:
                keypoint_list.extend([0,0])
            else:
                keypoint_list.extend([min(max(0,int(x)),self.width),min(max(0,int(y)),self.width)])
        return keypoint_list
            
    def get_keypoint_prompt(self, person_info:dict):
        # init seq with bbox
        box = np.array(person_info['box'],dtype=np.float64)
        box[0] = min(max(0,box[0]),self.width)/self.width
        box[1] = min(max(0,box[1]),self.height)/self.height
        box[2] = min(max(0,box[2]),self.width)/self.width
        box[3] = min(max(0,box[3]),self.height)/self.height
        box = self.quantize(box)

        seq = [self.BOS_code]
        seq.extend(list(map(int,box)))

        return seq
    
    def encode_segmentation(self, person_info: dict):
        # init seq with bbox
        box = np.array(person_info['box'],dtype=np.float64)
        box[0] = min(max(0,box[0]),self.width)/self.width
        box[1] = min(max(0,box[1]),self.height)/self.height
        box[2] = min(max(0,box[2]),self.width)/self.width
        box[3] = min(max(0,box[3]),self.height)/self.height
        box = self.quantize(box)

        # segmentation
        polys = person_info['segmentation']     
        
        poly_list = []
        for poly in polys:
            np.roll(polys,np.random.randint(0,poly.shape[0]),axis=0) # random select start point
            poly = self.negative_solver(poly[:self.max_num_point].ravel())
            assert len(poly) % 2 == 0,'a poly must have 2*n coords(n points)'
            poly[::2] = self.quantize(poly[::2]/self.width)
            poly[1::2] = self.quantize(poly[1::2]/self.height)
            poly_list.extend(poly)
            poly_list.append(self.seperate_token)
        poly_list = poly_list[:-1]
 
        seq = [self.BOS_code]
        seq.extend(list(map(int,box)))
        seq.extend(list(map(int, poly_list)))
        seq.append(self.EOS_code)

        init_len = len(box)        
        return seq, init_len

    def decode_segmentation(self, tokens):
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[5:-1] # jump bos,eos and box prompt

        polys = []
        split_indices = torch.nonzero(tokens == self.seperate_token).squeeze().tolist()
        split_indices.insert(0,-1) # append -1
        split_indices.append(len(tokens)) # append len(tokens)
        for idx in range(len(split_indices)-1):
            polys.append(tokens[split_indices[idx]+1:split_indices[idx+1]])    
        
        poly_point_list = []
        for poly in polys:
            # assert poly.shape[0] % 2 == 0, 'poly coord number should be even'
            poly_point_list.append(poly[:poly.size(0)-poly.size(0)%2].reshape([-1,2]).tolist())

        return poly_point_list

        
            
    def get_segmentation_prompt(self, person_info:dict):
        # init seq with bbox
        box = np.array(person_info['box'],dtype=np.float64)
        box[0] = min(max(0,box[0]),self.width)/self.width
        box[1] = min(max(0,box[1]),self.height)/self.height
        box[2] = min(max(0,box[2]),self.width)/self.width
        box[3] = min(max(0,box[3]),self.height)/self.height
        box = self.quantize(box)

        seq = [self.BOS_code]
        seq.extend(list(map(int,box)))

        return seq
    

import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_model


def initialize_model(p, checkpoint_path):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckp['model']
    n_state_dict = {}
    for k, v in state_dict.items():
        n_state_dict[k[7:]] = v
    model = get_model(p)
    model.load_state_dict(n_state_dict)
    model = model.cuda()
    print("model initialized..")
    return model

class DirectResize:
    """Resize samples so that the max dimension is the same as the giving one. The aspect ratio is kept.
    """

    def __init__(self, size):
        self.size = size

        self.mode = {
            'image': cv2.INTER_LINEAR
        }


    def resize(self, key, ori):
        new = cv2.resize(ori, self.size[::-1], interpolation=self.mode[key])

        return new

    def __call__(self, sample):
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.resize(key, val)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_infer_transforms(p):
    from data import transforms
    import torchvision
    dims = (3, 512, 512)
    valid_transforms = torchvision.transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.MaxResize(max_dim=512),
        DirectResize(dims[-2:]),
        # transforms.PadImage(size=dims[-2:]),
        transforms.ToTensor(),
    ])
    return valid_transforms

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def vis_semseg(_semseg):
    if False: # NYUD
        #Note: We ignore the background class as other related works. # nyud
        _semseg += 1 
        new_cmap = labelcolormap(41)
    else:
        # for pascal context, don't ignore background
        new_cmap = labelcolormap(21)
    _semseg = new_cmap[_semseg]  # skip the backgournd class
    return _semseg

def vis_parts(_semseg):
    if False: # NYUD
        #Note: We ignore the background class as other related works. # nyud
        _semseg += 1 
        new_cmap = labelcolormap(41)
    else:
        # for pascal context, don't ignore background
        new_cmap = labelcolormap(7)
    _semseg = new_cmap[_semseg]  # skip the backgournd class
    return _semseg

import torch.nn.functional as F
@torch.no_grad()
def get_predictions(task, preds, meta):
    for idx, pred in enumerate(preds):
        im_height = meta['im_size'][1]#[idx]
        im_width = meta['im_size'][0]#[idx]

        pred = pred.unsqueeze(0)
        pred = F.interpolate(pred, (im_height, im_width), mode='bilinear')

        if task in ['edge']:
            pred = 255 * torch.sigmoid(pred.squeeze(1))
        elif task in ['semseg', 'human_parts']:
            pred = torch.argmax(pred, dim=1)
        elif task == 'normals':
            norm = torch.norm(pred, p='fro', dim=1, keepdim=True).expand_as(pred)
            pred = 255 * (pred.div(norm) + 1.0) / 2.0
            pred[norm == 0] = 0
        elif task == 'depth':
            pass
        elif task == 'sal':
            assert pred.shape[1] == 2
            # two class probabilites
            pred = nn.functional.softmax(pred, dim=1)[:, 1, :, :]*255
        else:
            raise ValueError
        pred = pred.squeeze()

        if pred.ndim == 3:
            pred = pred.permute(1, 2, 0) 
        arr = pred.cpu().numpy()
        # visualize result 
        import pdb
        if task == 'semseg':
            arr = vis_semseg(arr)
        elif task == 'sal':
            arr = arr[:, :, np.newaxis]
            arr = arr.repeat(3, axis=2)
        elif task == 'edge':
            arr = arr[:, :, np.newaxis]
            arr = arr.repeat(3, axis=2)
        elif task == 'human_parts':
            arr = vis_parts(arr)
        elif task == 'normals':
            pass
        elif task == 'depth':
            arr = arr.squeeze()
            return arr.astype(np.uint8)

        return arr.astype(np.uint8) #image

# Parser
parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--config_exp', default='./configs/pascal/pascal_vitLp16.yml',
                    help='Config file for the experiment')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--image_path',
                    help='Image path which has to be parsed')
parser.add_argument('--ckp_path', default=None,
                    help='Config file for the experiment')
parser.add_argument('--save_dir', default="../output",
                    help='Save output image')
# parser.add_argument('--task', default="human_parts",
#                     help='Save output image')
args = parser.parse_args()

@torch.no_grad()
def infer(task):
    p = create_config(args.config_exp,{'run_mode': 'infer'})
    
    checkpoint_path = args.ckp_path
    img = Image.open(args.image_path).convert('RGB')
    ori_size = img.size # (w, h)
    img = np.asarray(img, dtype=np.float32)
    img = {'image':img}
    valid_transforms = get_infer_transforms(p)
    inp = valid_transforms(img)
    inp = inp['image']
    img = np.array(inp.permute(1,2,0))
    inp.requires_grad = False
    inp = inp.unsqueeze(0).cuda()

    model = initialize_model(p, checkpoint_path)
    model.eval()
    with torch.no_grad():
        out = model(inp)

    # save prediction
    meta = {'im_size': ori_size}
    prediction = get_predictions(task, out[task], meta).squeeze()

    save_path = os.path.join(args.save_dir, task + '.png')
    mkdir_if_missing(args.save_dir)
    plt.imsave(save_path, prediction)
    print(f"Prediction saved at {save_path}.")
  
if __name__ == "__main__":
    # Run example: 
    # CUDA_VISIBLE_DEVICES=0 python inference.py --image_path=IMAGE_PATH --ckp_path=CKP_PATH --save_dir=SAVE_DIR

    # task = args.task
    tasks = ['semseg', 'normals', 'sal', 'edge', 'human_parts']
    for task in tasks:
        infer(task)

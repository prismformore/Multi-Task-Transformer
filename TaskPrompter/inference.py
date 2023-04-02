# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import torch
from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_model
from utils.visualization_utils import vis_pred_for_one_task

# stuttgart camera parameters 
bbox_camera_params = {
    "sensor_T_ISO_8855": torch.tensor([
        [
            0.9990881051503779,
            -0.01948468779721943,
            -0.03799085532693703,
            -1.6501524664770573
        ],
        [
            0.019498764210995674,
            0.9998098810245096,
            0.0,
            -0.1331288872611436
        ],
        [
            0.03798363254444427,
            -0.0007407747301939942,
            0.9992780868764849,
            -1.2836173638418473
        ]
    ]),
    "fx": torch.tensor(2262.52),
    "fy": torch.tensor(2265.3017905988554),
    "u0": torch.tensor(1096.98),
    "v0": torch.tensor(513.137),
}
def get_K_matrix(bbox_camera_params):
    fx, u0, fy, v0 = bbox_camera_params["fx"], bbox_camera_params["u0"], bbox_camera_params["fy"], bbox_camera_params["v0"],
    K_matrix = np.zeros((3, 3))
    K_matrix[0][0] = fx
    K_matrix[0][2] = u0
    K_matrix[1][1] = fy
    K_matrix[1][2] = v0
    K_matrix[2][2] = 1
    return torch.tensor(K_matrix)

K_matrix = get_K_matrix(bbox_camera_params)
bbox_camera_params = {k: [v] for k, v in bbox_camera_params.items()}

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
    if p.train_db_name == 'PASCALContext':
        dims = (3, 512, 512)
        infer_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.MaxResize(max_dim=512),
            DirectResize(dims[-2:]),
            # transforms.PadImage(size=dims[-2:]),
            transforms.ToTensor(),
        ])
    elif p.train_db_name == 'Cityscapes3D':
        dims = (3, 1024, 2048)
        infer_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            DirectResize(dims[-2:]),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    return infer_transforms


@torch.no_grad()
def infer_one_image(image_path):
    bs = 1
    p = create_config(args.config_path, {'run_mode': 'infer'})
    if p.train_db_name == 'PASCALContext':
        tasks = ['semseg', 'normals', 'sal', 'edge', 'human_parts']
    elif p.train_db_name == 'Cityscapes3D': 
        tasks = ['semseg', 'depth', '3ddet']

    else:
        raise NotImplementedError
    
    checkpoint_path = args.ckp_path
    img_name = image_path.split('/')[-1].split('.')[0]
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_size = img.shape[:2] # (h, w, 3)
    img = {'image':img}
    infer_transforms = get_infer_transforms(p)
    inp = infer_transforms(img)
    inp = inp['image']
    inp = inp.unsqueeze(0).cuda()

    model = initialize_model(p, checkpoint_path)
    model.eval()
    output = model(inp)

    # save prediction
    if p.train_db_name == 'PASCALContext':
        meta = {'img_name': [img_name],
                'img_size': [ori_size]}
        sample = {'meta': meta}
    elif p.train_db_name == 'Cityscapes3D': 
        meta = {'img_name': [img_name],
                 'img_size': [ori_size],
                 'K_matrix': [K_matrix],
                 'scale_factor': [np.array([1, 1])],
                 }
        
        # 
        sample = {'meta': meta, 'bbox_camera_params': bbox_camera_params}

    sample['image'] = inp
    for task in tasks:
        task_save_dir = os.path.join(args.save_dir, task)
        mkdir_if_missing(task_save_dir)
        vis_pred_for_one_task(p, sample, output, task_save_dir, task=task)

# Parser
parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--config_path', default='./configs/pascal/pascal_vitLp16.yml',
                    help='Config file for the experiment')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--image_path',
                    help='Image path which has to be parsed')
parser.add_argument('--ckp_path', default=None,
                    help='Config file for the experiment')
parser.add_argument('--save_dir', default="../output",
                    help='Save output image')

args = parser.parse_args()
  
if __name__ == "__main__":
    # Run example: 
    # CUDA_VISIBLE_DEVICES=0 python inference.py --config_path=CONFIG_PATH --image_path=IMAGE_PATH --ckp_path=CKP_PATH --save_dir=SAVE_DIR

    infer_one_image(args.image_path)

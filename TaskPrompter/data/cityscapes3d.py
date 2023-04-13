# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import json
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation
import torch.utils.data as data
from PIL import Image, ImageDraw, ImageFont
import imageio, cv2
import torch
import copy
import torchvision.transforms as t_transforms
import pickle

from cityscapesscripts.helpers.annotation import CsBbox3d
from cityscapesscripts.helpers.box3dImageTransform import (
    Camera, 
    Box3dImageTransform,
    CRS_V,
    CRS_C,
    CRS_S
)


evalLabels = ["car", "truck", "bus", "train", "motorcycle", "bicycle"] # totally 6 clases for 3d detection
evalLabelsDict = {na: i for i, na in enumerate(evalLabels)}

def det_cls_str2no(cls):
    if cls in evalLabels:
        return evalLabelsDict[cls]
    else:
        raise NotImplementedError

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def imresize(img, size, mode, resample):
    size = (size[1], size[0]) # width, height
    _img = Image.fromarray(img)#, mode=mode)
    _img = _img.resize(size, resample)
    _img = np.array(_img)
    return _img

class CITYSCAPES3D(data.Dataset):
    def __init__(self, p, root, split=["train"], is_transform=False,
                 img_size=[1024, 2048], augmentations=None, task_list=['semseg', 'depth', '3ddet'], ignore_index=255):

        if isinstance(split, str):
            split = [split]
        else:
            split.sort()
            split = split

        self.split = split
        self.root = root
        self.split_text = '+'.join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size 
        self.dd_label_map_size = p.dd_label_map_size

        self.task_flags = {'semseg': True, 'insseg': False, 'depth': True}
        self.task_list = task_list
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, 'leftImg8bit', _split)
            self.annotations_base = os.path.join(self.root, 'gtFine', _split)
            self.files[self.split_text] += recursive_glob(rootdir=self.images_base, suffix='.png')
            self.depth_base = os.path.join(self.root, 'disparity',  _split)
            self.camera_base = os.path.join(self.root, 'camera',  _split) 
            self.det_base = os.path.join(self.root, 'gtBbox3d',  _split) 
        ori_img_no = len(self.files[self.split_text])

        if self.split_text == 'train' and True:
            # remove samples without detection labels
            self.find_bad_samples()
            print("Found %d %s images, %d are used (with 3d bbox annotation)" % (ori_img_no, self.split_text, len(self.files[self.split_text])))
        else:
            print("Found %d %s images, %d are used (may not have 3d bbox annotation)" % (ori_img_no, self.split_text, len(self.files[self.split_text])))

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = ignore_index
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.ori_img_size = [1024, 2048]
        self.label_dw_ratio = img_size[0] / self.ori_img_size[0] # hacking

        if len(self.files[self.split_text]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split_text, self.images_base))

        # image to tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_transform = t_transforms.Compose([t_transforms.ToTensor(), t_transforms.Normalize(mean, std)])


    def __len__(self):
        return len(self.files[self.split_text])

    def __getitem__(self, index):
        
        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        instance_path = os.path.join(self.annotations_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')
        depth_path = os.path.join(self.depth_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'disparity.png')  
        camera_path = os.path.join(self.camera_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'camera.json')  
        det_path = os.path.join(self.det_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'gtBbox3d.json')  
                            

        img = cv2.imread(img_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = {'image': img}
        sample['meta'] = {'img_name': img_path.split('.')[0].split('/')[-1],
                        'img_size': (img.shape[0], img.shape[1]),
                        'dd_label_map_size': self.dd_label_map_size,
                        'scale_factor': np.array([self.img_size[1]/img.shape[1], self.img_size[0]/img.shape[0]]), # in xy order
                        }

        if 'semseg' in self.task_list:
            lbl = imageio.imread(lbl_path)
            sample['semseg'] = lbl

        if 'depth' in self.task_list:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) # disparity

            depth[depth>0] = (depth[depth>0] - 1) / 256 # disparity values

            # make the invalid idx to -1
            depth[depth==0] = -1

            # assign the disparity of sky to zero
            sky_mask = lbl == 10
            depth[sky_mask] = 0

            if False:
                # The model directly regress the depth value instead of disparity. Based on the official implementation: https://github.com/mcordts/cityscapesScripts/issues/55
                camera = json.load(open(camera_path))
                depth[depth>0] = camera["extrinsic"]["baseline"] * camera["intrinsic"]["fx"] / depth[depth>0] # real depth
            sample['depth'] = depth

        if 'insseg' in self.task_list:
            ins = imageio.imread(instance_path)

        if '3ddet' in self.task_list:
            # get 2D/3D detection labels
            det_labels, K_matrix, bbox_camera_params = self.load_det(det_path) 
            sample['bbox_camera_params'] = bbox_camera_params
            sample['det_labels'] = det_labels
            sample['det_label_number'] = len(det_labels)
            sample['meta']['K_matrix'] = K_matrix

        if 'insseg' in self.task_list:
            sample['ins'] = ins

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        if 'semseg' in self.task_list:
            sample['semseg'] = self.encode_segmap(sample['semseg'])
        
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def transform(self, sample):
        img = sample['image']
        if 'semseg' in self.task_list:
            lbl = sample['semseg']
        if 'depth' in self.task_list:
            depth = sample['depth']

        img_ori_shape = img.shape[:2]
        img = img.astype(np.uint8)

        if self.img_size != self.ori_img_size:
            img = imresize(img, (self.img_size[0], self.img_size[1]), 'RGB', Image.BILINEAR)

        if 'semseg' in self.task_list:
            classes = np.unique(lbl)
            lbl = lbl.astype(float)
            # if self.img_size != self.ori_img_size:
            if self.dd_label_map_size != self.ori_img_size:
                lbl = imresize(lbl, (int(self.dd_label_map_size[0]), int(self.dd_label_map_size[1])), 'F', Image.NEAREST) # TODO(ozan) /8 is quite hacky
            lbl = lbl.astype(int)

        if 'depth' in self.task_list:
            # if self.img_size != self.ori_img_size:
            if self.dd_label_map_size != self.ori_img_size:
                depth = imresize(depth, (int(self.dd_label_map_size[0]), int(self.dd_label_map_size[1])), 'F', Image.NEAREST)
                # depth = depth * self.label_dw_ratio
            depth = np.expand_dims(depth, axis=0)
            depth = torch.from_numpy(depth).float()
            sample['depth'] = depth

        if 'semseg' in self.task_list:
            if not np.all(np.unique(lbl[lbl!=self.ignore_index]) < self.n_classes):
                print('after det', classes,  np.unique(lbl))
                raise ValueError("Segmentation map contained invalid class values")
            lbl = torch.from_numpy(lbl).long()
            sample['semseg'] = lbl

        img = self.img_transform(img)
        sample['image'] = img

        return sample

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask==_voidc] = self.ignore_index
        old_mask = mask.copy()
        for _validc in self.valid_classes:
            mask[old_mask==_validc] = self.class_map[_validc] 
        return mask

    def find_bad_samples(self,):
        '''
        Find samples without detection labels and remove them from the self.files
        '''
        if False: # load from saved bad sample list
            split = self.split[0]
            paths = ['data/bad_samples_' + split + '.pickle', 'bad_samples_' + split + '.pickle']
            for pa in paths:
                if os.path.exists(pa):
                    pkl_path = pa
            with open(pkl_path, 'rb') as f:
                bad_samples = pickle.load(f) 
            for bad_sa in bad_samples:
                self.files[split].remove(bad_sa)
            return

        bad_samples = []
        split = self.split[0]
        tmp_files = copy.deepcopy(self.files)

        for index in range(len(self)):
            img_path = self.files[self.split_text][index].rstrip()
            det_path = os.path.join(self.det_base,
                                        img_path.split(os.sep)[-2],
                                        os.path.basename(img_path)[:-15] + 'gtBbox3d.json')  

            bbox_json = json.load(open(det_path))
            obj_number = len(bbox_json["objects"])
            valid_obj_number = copy.deepcopy(obj_number)

            for i in range(obj_number):
                if bbox_json["objects"][i]["label"] not in evalLabels:
                    valid_obj_number -= 1
                    continue

            if valid_obj_number == 0:
                bad_samples.append(img_path)
                # remove bad samples
                tmp_files[split].remove(img_path)

        self.files = tmp_files

        with open('CS3D_bad_samples_' + split + '.txt', 'w') as f:
            f.write(str(bad_samples))
        with open('CS3D_bad_samples_' + split + '.pickle', 'wb') as f:
            pickle.dump(bad_samples, f)

    def load_det(self, det_path):
        # get 2D/3D detection labels
        bbox_json = json.load(open(det_path))
        bbox_camera_params = {'fx': np.array(bbox_json["sensor"]["fx"], dtype=np.float32),
                              'fy': np.array(bbox_json["sensor"]["fy"], dtype=np.float32), 
                              'u0': np.array(bbox_json["sensor"]["u0"], dtype=np.float32),
                              'v0': np.array(bbox_json["sensor"]["v0"], dtype=np.float32),
                              'sensor_T_ISO_8855': np.array(bbox_json["sensor"]["sensor_T_ISO_8855"], dtype=np.float32)
                              }
        bbox_camera = Camera(fx=bbox_camera_params["fx"],
                            fy=bbox_camera_params["fy"],
                            u0=bbox_camera_params["u0"],
                            v0=bbox_camera_params["v0"],
                            sensor_T_ISO_8855=bbox_camera_params["sensor_T_ISO_8855"])

        # Create a CsBox3d object for the 3D annotation
        obj_number = len(bbox_json["objects"])

        K_matrix = np.float32(get_projection_matrix(bbox_json))

        det_labels = []
        for i in range(obj_number):
            if bbox_json["objects"][i]["label"] not in evalLabels:
                continue

            # Create the Box3dImageTransform object
            box3d_annotation = Box3dImageTransform(camera=bbox_camera)
            obj = CsBbox3d()
            obj.fromJsonText(bbox_json["objects"][i])
            box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V) # the annotation is given in coordinate system V, it must be transformed from V → C → S → I.

            # Similar to the box vertices, you can retrieve box parameters center, size and rotation in any coordinate system
            size_S, center_S, rotation_S = box3d_annotation.get_parameters(coordinate_system=CRS_S)

            # project center from S to I to check consistency.  V -> C -> S -> I
            size_S = np.array(size_S, dtype=np.float32)
            center_S = np.array(center_S, dtype=np.float32)
            center_2d = np.matmul(K_matrix, center_S)
            depth = center_2d[-1]
            center_2d = center_2d[:2] / depth
            center_I = np.concatenate([center_2d, depth[None]])

            # get yaw, pitch and roll from rotation_S (Quaternion)
            # rotation_S_e = np.array(rotation_S.yaw_pitch_roll, dtype=np.float32) # range: [-pi, pi]
            rot = Rotation.from_quat(rotation_S.q)
            rotation_S = rot.as_euler('ZXY') #order: [pitch, roll, yaw]
            rotation_S = np.float32(rotation_S)

            # generate amodal 2D box from these values (including the occluded part)
            [xmin, ymin, xmax, ymax] = obj.bbox_2d.bbox_amodal 
            bbox_amodal = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            bbox_modal = np.array(obj.bbox_2d.bbox_modal, dtype=np.float32)

            det_labels.append({
                                "size_S": size_S,
                                "center_S": center_S,
                                "center_I": center_I,
                                "rotation_S": rotation_S,
                                "bbox_amodal": bbox_amodal,
                                "bbox_modal": bbox_modal,
                                'label': det_cls_str2no(obj.label), 
                                })
        return det_labels, K_matrix, bbox_camera_params

def get_projection_matrix(bbox_json):
    fx, u0, fy, v0 = bbox_json["sensor"]["fx"], bbox_json["sensor"]["u0"], bbox_json["sensor"]["fy"], bbox_json["sensor"]["v0"],
    K_matrix = np.zeros((3, 3))
    K_matrix[0][0] = fx
    K_matrix[0][2] = u0
    K_matrix[1][1] = fy
    K_matrix[1][2] = v0
    K_matrix[2][2] = 1
    return K_matrix

class ComposeAug(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        sample['image'], sample['semseg'], sample['depth'] =  np.array(sample['image'], dtype=np.uint8), np.array(sample['semseg'], dtype=np.uint8), np.array(sample['depth'], dtype=np.float32)
        sample['image'], sample['semseg'], sample['depth'] = Image.fromarray(sample['image'], mode='RGB'), Image.fromarray(sample['semseg'], mode='L'), Image.fromarray(sample['depth'], mode='F')
        if 'insseg' in sample.keys():
            sample['insseg'] = np.array(sample['insseg'], dtype=np.int32)
            sample['insseg'] = Image.fromarray(sample['insseg'], mode='I')

        assert sample['image'].size == sample['semseg'].size
        assert sample['image'].size == sample['depth'].size
        if 'insseg' in sample.keys():
            assert sample['image'].size == sample['insseg'].size

        for a in self.augmentations:
            sample = a(sample)

        sample['image'] = np.array(sample['image'])
        sample['semseg'] = np.array(sample['semseg'], dtype=np.uint8)
        sample['depth'] = np.array(sample['depth'], dtype=np.float32)
        if 'insseg' in sample.keys():
            sample['insseg'] = np.array(sample['insseg'], dtype=np.uint64)

        return sample
    
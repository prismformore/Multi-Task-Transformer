[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-2d-3d-multi-task-learning-on-cityscapes/monocular-depth-estimation-on-cityscapes-3d)](https://paperswithcode.com/sota/monocular-depth-estimation-on-cityscapes-3d?p=joint-2d-3d-multi-task-learning-on-cityscapes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-2d-3d-multi-task-learning-on-cityscapes/3d-object-detection-on-cityscapes-3d)](https://paperswithcode.com/sota/3d-object-detection-on-cityscapes-3d?p=joint-2d-3d-multi-task-learning-on-cityscapes)
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)


<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/14089338/229339707-5da4ec14-8fa5-44e4-bd73-d0859cdc7747.png" alt="taskprompter" style="width: 60%; min-width: 300px; display: block; margin: auto; background-color: transparent;">
</p>


# :fire: [ICLR2023] [TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding](https://openreview.net/pdf?id=-CwPopPJda)

<p align="center">
  <img alt="img-name" src="https://user-images.githubusercontent.com/14089338/228763472-04bdacf2-cd7d-47b8-8afb-f2c8a52a96d5.gif" width="900">
  <br>
    <em>Joint 2D-3D multi-task scene understanding on Cityscapes-3D: 3D detection, segmentation, and depth estimation.</em>
</p>

<p align="center">
    <a href="https://openreview.net/pdf?id=-CwPopPJda">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://youtu.be/-eAvl8CLV1g">Demo</a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2304.00971">Cityscapes-3D Supp. (arXiv)</a>
  
</p>  

##  :scroll: Introduction
This repository contains the codes and models for TaskPrompter, our multi-task prompting transformer model, which has been implemented on three datasets: Cityscapes-3D (NEW!), PASCAL-Context, and NYUD-v2. 

- TaskPrompter simultaneously models task-generic and task-specific representations, as well as cross-task representation interactions in a single module. 
- Its compact design establishes a new state-of-the-art (SOTA) performance while reducing computation costs on PASCAL-Context and NYUD-v2. 
- Additionally, we propose a new joint 2D-3D multi-task learning benchmark based on Cityscapes-3D,  allowing TaskPrompter to produce predictions for 3D detection, segmentation, and depth estimation utilizing a single model, training, and inference. Our 3D detection achievements significantly exceed the prior best single-task outcomes on Cityscapes-3D. 

Please check the ICLR 2023 [paper](https://openreview.net/pdf?id=-CwPopPJda) for more details.
<p align="center">
  <img alt="img-name" src="https://user-images.githubusercontent.com/14089338/228757124-2b0d3272-6c62-4271-81be-ce29ebc6ebab.png" width="800">
  <br>
    <em>Framework overview of the proposed TaskPrompter for multi-task scene understanding.</em>
</p>


# :tv: Demo
<table class="center">
<tr>
  <td><img src="https://user-images.githubusercontent.com/14089338/229057421-e89fe102-988f-4faa-ab43-38dd0224f33c.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/14089338/229057434-d887417c-805a-4a73-bbba-1d18a3f18baf.gif"></td>
</tr>
<tr>
    <td><img src="https://user-images.githubusercontent.com/14089338/229057445-c9ab7ac1-f898-4b99-a579-287a0f3986e8.gif"></td>
    <td><img src="https://user-images.githubusercontent.com/14089338/229059776-611611ff-dd14-41a9-964d-1258a5a7c478.gif"></td>
</tr>
</table>


To qualitatively demonstrate the powerful multi-task performance of TaskPrompter, we visualizes its predictions on Cityscapes-3D. 
**Watch the full version of demo [here](https://youtu.be/-eAvl8CLV1g)!**

# News
:triangular_flag_on_post: **Updates** 
- :white_check_mark: April, 2023: Update codes and model checkpoints for Cityscapes-3D. Now we evaluate disparity instead of GT depth, which is a more popular setting on Cityscapes. We also slightly improve the model for higher efficiency and performance. 
- :white_check_mark: April, 2023: Release TaskPrompter models trained on Cityscapes-3D, PASCAL-Context, and NYUD-v2 dataset!


# :grinning: Train your **TaskPrompter**!

## 1. Build recommended environment
We inherit the environement of InvPT, and here is a successful path to deploy it:
```bash
conda create -n taskprompter python=3.7
conda activate taskprompter
pip install tqdm Pillow easydict pyyaml imageio scikit-image tensorboard
pip install opencv-python==4.5.4.60 setuptools==59.5.0

# Example of installing pytorch-1.10.0 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.5.4 einops==0.4.1
```

### Additional environment setup for 3D detection on Cityscapes-3D
If you would like to set up the multi-task Cityscapes-3D benchmark with 3D detection, you will also need to install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md). MMDetection3D is highly dependent on the versions of related packages. Here is our suggested installation path:
```bash
pip install openmim
mim install mmcv-full==1.6.2
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
```
Then, install the MMDetection3D package in another directory you want:
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout 1.0  # 1.0.0rc6
pip install -e .
```

Next, come back to this repository and run the setup script for the NMS tool written in CUDA (borrowed from MMDetection3D):
```bash
bash setup_3ddet.sh
```

For evaluation on Cityscapes-3D, you will also need to install the official evaluation tool:
```bash
pip install cityscapesscripts
```


## 2. Get data
### Cityscapes-3D
To obtain the necessary files for multi-task learning of 3D detection, semantic segmentation, and depth estimation, please sign up on the official [Cityscapes-3D website](https://www.cityscapes-dataset.com/) and download the required files.
```
leftImg8bit_trainvaltest.zip     - 5000 images with fine annotations
gtFine_trainvaltest.zip          - semantic segmentation annotations, we use 19 classes
gtBbox3d_trainvaltest.zip        - 3D bounding box for 3D detection
disparity_trainvaltest.zip       - disparity maps for later computing depth maps in dataloader
camera_trainvaltest.zip          - camera parameters
```

Unzip the files and puth them in the same directory. 
To confirm that you have placed the files in the correct position, you can check our dataloader at ```data/cityscapes3d.py```.


### PASCAL-Context and NYUD-v2
You can download PASCAL-Context and NYUD-v2 from ATRC's repository:
```bash
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
```
And then extract the datasets by:
```bash
tar xfvz NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
```

**You need to put the datasets into one directory and specify the directory as ```db_root``` variable in ```configs/mypath.py```.**


## 3. Train the model
The config files are defined in ```./configs```, the output directory is also defined in your config file.

Before start training, you need to change the ```.sh``` files for different configuation. We use [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for multi-gpu training by default. You may need to read realted documents before setting the gpu numbers. 

Cityscapes-3D
```bash
bash run_taskprompter_cs3d.sh
```
PASCAL-Context:
```bash
bash run_taskprompter_pascal.sh
```

NYUD-v2
```bash
bash run_taskprompter_nyud.sh
```


All models are defined in ```models/``` so it should be easy to **deploy your own model in this framework**.

## 4. Evaluate the model
The training script itself includes evaluation. 
For inferring with pre-trained models, you need to change ```run_mode``` in ```run*.sh``` to ```infer```.

### **Special evaluation for boundary detection**
We follow previous works and use Matlab-based [SEISM](https://github.com/jponttuset/seism) project to compute the optimal dataset F-measure scores. The evaluation code will save the boundary detection predictions on the disk. 

Specifically, identical to ATRC and ASTMT, we use [maxDist](https://github.com/jponttuset/seism/blob/6af0cad37d40f5b4cbd6ca1d3606ec13b176c351/src/scripts/eval_method.m#L34)=0.0075 for PASCAL-Context and maxDist=0.011 for NYUD-v2. Thresholds for HED (under seism/parameters/HED.txt) are used. ```read_one_cont_png``` is used as IO function in SEISM.

# :partying_face:	 Pre-trained models
To faciliate the community to reproduce our SoTA results, we re-train our best performing models with the training code in this repository and provide the weights for the reserachers.

### Download pre-trained models
|Version | Dataset | Download | 3D det (mDS) | Depth (RMSE) | Segmentation (mIoU) |  Human parsing (mIoU) | Saliency (maxF) | Normals (mErr) | Boundary (odsF) | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TaskPrompter (Swin-B)| Cityscapes-3D | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EaryGO5Z_vdGuVUEVgIrKNQBm_f80765fJKpjo8Bd_YJiA?e=uPlG0t) | 32.36 | 4.67 | 79.38 | - | - | - | - |
| TaskPrompter (ViT-L)| PASCAL-Context | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EYX13WbZOPRHnsV5fPeolEYBeafey2PYnTpd7jLFlGDc5w?e=Yg4enV) | - | - |80.79 | 68.81 |84.26 | 13.58 | 73.80 |
| TaskPrompter (ViT-B)| PASCAL-Context | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/Ea4nDFZaJwhDvllsHt81ZJsBkvmuBBpT8JZGiXUon7xScA?e=QsWoSj) | - | - | 78.63 | 66.93 |84.77 | 13.42 | 73.60 |
| TaskPrompter (ViT-L) | NYUD-v2 | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EQx8Ub49_ahKnDQJEv9Sjz8BrhQ33GyFHaas8eQaq0uVcA?e=DnT2zg) | - | 0.5062 | 55.90 | - | - | 18.58 | 77.70 |

### Infer with the pre-trained models
Simply set the pre-trained model path in ```run.sh``` by adding ```--trained_model MODEL_PATH```.
You also need to change ```run_mode``` in ```run.sh``` to ```infer```.

### Generate multi-task predictions form any image
To generate multi-task predictions from an image with the pre-trained model on PASCAL-Context and Cityscapes-3D (only support camera parameters used in "stuttgart" of Cityscapes), please use ```inference.py```. An example running script is:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --config_path=CONFIG_PATH --image_path=IMAGE_PATH --ckp_path=CKP_PATH --save_dir=SAVE_DIR
```

For tools to visualize the predictions, please check ```utils/visualization_utils.py```.


# :hugs: Cite
BibTex:
```
@InProceedings{taskprompter2023,
  title={TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding},
  author={Ye, Hanrong and Xu, Dan},
  booktitle={ICLR},
  year={2023}
}
```
Please consider :star2: star our project to share with your community if you find this repository helpful!

# :blush: Contact
Please contact [Hanrong Ye](https://sites.google.com/site/yhrspace/) if any questions.

# :+1: Acknowledgement
This repository is built upon InvPT. We borrow partial codes from [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

# :business_suit_levitating: License
[Creative commons license](http://creativecommons.org/licenses/by-nc/4.0/) which allows for personal and research use only. 

For commercial useage, please contact the authors. 

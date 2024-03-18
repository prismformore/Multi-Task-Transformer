# :fire: [CVPR2024] [DiffusionMTL: Learning Multi-Task Denoising Diffusion Model from Partially Annotated Data](https://openreview.net/pdf?id=-CwPopPJda)

<p align="center">
    <a href="https://openreview.net/pdf?id=-CwPopPJda">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://youtu.be/lDnbkM3EwVM">Demo</a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2304.00971">Cityscapes-3D Supp. (arXiv)</a>
  
</p>  

##  :scroll: Introduction
This repository contains the codes and models for DiffusionMTL, our multi-task scene understanding model trained with partially-annotated data. 

- We reformulate the partially-labeled multi-task dense prediction as a pixel-level denoising problem.
- To exploit multi-task consistency in denois- ing, we further introduce a Multi-Task Conditioning strategy, which can implicitly utilize the complementary nature of the tasks to help learn the unlabeled tasks, leading to an improvement in the denoising performance of the different tasks.
- State-of-the-art performance on three benchmarks under two different partial-labeling evaluation settings.

Please check the CVPR 2024 [paper](https://openreview.net/pdf?id=-CwPopPJda) for more details.
<p align="center">
  <img alt="img-name" src="https://user-images.githubusercontent.com/14089338/228757124-2b0d3272-6c62-4271-81be-ce29ebc6ebab.png" width="800">
  <br>
    <em>Framework overview of the proposed TaskPrompter for multi-task scene understanding.</em>
</p>


To qualitatively demonstrate the powerful multi-task performance of TaskPrompter, we visualizes its predictions on Cityscapes-3D. 
**Watch the full version of demo [here](https://youtu.be/lDnbkM3EwVM)!**

# News
:triangular_flag_on_post: **Updates** 
- :white_check_mark: March, 2024: Release codes for feature diffusion training on PASCAL dataset!


# :grinning: Train your model!

## 1. Build recommended environment
We inherit the environement of TaskPrompter, and here is a successful path to deploy it:
```bash
conda create -n taskprompter python=3.7
conda activate taskprompter
pip install tqdm Pillow easydict pyyaml imageio scikit-image tensorboard termcolor matplotlib
pip install opencv-python==4.5.4.60 setuptools==59.5.0

# Example of installing pytorch-1.10.0 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.5.4 einops==0.4.1
```


## 2. Get data
### PASCAL-Context and NYUD-v2
We use the same data (PASCAL-Context and NYUD-v2) as ATRC. You can download the data from:
[PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab),
[NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c)

And then extract the datasets by:
```bash
tar xfvz NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
```

**You need to put the datasets into one directory and specify the directory as ```db_root``` variable in ```configs/mypath.py```.**


## 3. Train the model
The config files are defined in ```./configs```.

Before start training, you need to change the ```.sh``` files for different configuation. We use [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for multi-gpu training by default. You may need to read realted documents before setting the gpu numbers. 

PASCAL-Context:
```bash
bash run.sh
```

# :partying_face:	 Pre-trained models
To faciliate the community to reproduce our SoTA results, we re-train our best performing models with the training code in this repository and provide the weights for the reserachers.

### Download pre-trained models
|Version | Dataset | Download | Segmentation (mIoU) |  Human parsing (mIoU) | Saliency (maxF) | Normals (mErr) | Boundary (odsF) | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DiffusionMTL (Feature Diffusion)| PASCAL-Context (one-label) | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EYX13WbZOPRHnsV5fPeolEYBeafey2PYnTpd7jLFlGDc5w?e=Yg4enV) | 57.16 | 59.28 |78.00 | 16.17 | 64.60 |

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
Please consider :star2: star our project to share with your community if you find this repository helpful!

BibTex:
```
@InProceedings{diffusionmtl,
  title={DiffusionMTL: Learning Multi-Task Denoising Diffusion Model from Partially Annotated Data},
  author={Ye, Hanrong and Xu, Dan},
  booktitle={CVPR},
  year={2024}
}
```

# :blush: Contact
Please contact [Hanrong Ye](https://sites.google.com/site/yhrspace/) if any questions.
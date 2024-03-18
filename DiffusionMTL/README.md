# :fire: [CVPR2024] [DiffusionMTL: Learning Multi-Task Denoising Diffusion Model from Partially Annotated Data](https://openreview.net/pdf?id=-CwPopPJda)

<p align="center">
    <a href="https://openreview.net/pdf?id=-CwPopPJda">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;
</p>  

##  :scroll: Introduction
This repository contains the codes and models for DiffusionMTL, our multi-task scene understanding model trained with partially annotated data. 

- We reformulate the partially-labeled multi-task dense prediction as a pixel-level denoising problem.
- To exploit multi-task consistency in denois- ing, we further introduce a Multi-Task Conditioning strategy, which can implicitly utilize the complementary nature of the tasks to help learn the unlabeled tasks, leading to an improvement in the denoising performance of the different tasks.
- State-of-the-art performance on three benchmarks under two different partial-labeling evaluation settings.

Please check the CVPR 2024 [paper](https://openreview.net/pdf?id=-CwPopPJda) for more details.
<p align="center">
  <img alt="img-name" src="https://github.com/prismformore/Multi-Task-Transformer/assets/14089338/5862c11f-cd1b-464c-b04e-28a729dde7d4" width="600">
  <br>
    <em>Framework overview of the proposed TaskPrompter for multi-task scene understanding.</em>
</p>


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

Edge evaluation code: https://github.com/prismformore/Boundary-Detection-Evaluation-Tools

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
| DiffusionMTL (Feature Diffusion)| PASCAL-Context (one-label) | [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ERxytOgnrZpBhBkaJNdBTlUBHNKu7E92MHeNRb7jQshhuw?e=wCCle4) | 57.16 | 59.28 |78.00 | 16.17 | 64.60 |

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

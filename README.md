<div align="center">
    <h2>
        Context-Aggregated and SAM-Guided Network for ViT based Instance Segmentation in Remote Sensing Images
    </h2>
</div>


## Introduction

This repository is the code implementation of the paper Context-Aggregated and SAM-Guided Network for ViT based Instance Segmentation in Remote Sensing Images, which is based on the [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main) project.

The current branch has been tested under PyTorch 1.9 and CUDA 11.1, supports Python 3.7+, and is compatible with most CUDA versions.


## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.8
- PyTorch 1.9 or higher, recommended 1.9
- CUDA 11.1 or higher, recommended 11.1
- MMCV 2.0 or higher, recommended 2.0.1

### Environment Installation

We recommend using Miniconda for installation. The following command will create a virtual environment named `csnet` and install PyTorch and MMCV.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `csnet` and activate it.

```shell
conda create -n csnet python=3.8 -y
conda activate csnet
```

**Step 2**: Install [PyTorch](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --index-url https://download.pytorch.org/whl/cu111
```
Or

```shell
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 pytorch-cuda=11.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV2.0.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.0.1
```

**Step 4**: Install other dependencies. Please refer to the installation requirements of [SAM](https://github.com/facebookresearch/segment-anything) and [MMDetection](https://github.com/open-mmlab/mmdetection).




</details>

### Install CSNet

Download or clone the CSNet repository.


```shell
git clone git@github.com:LiuShuangZhou/CSNet.git
cd CSNet
```

## Dataset Preparation

<details>

### Basic Instance Segmentation Dataset

We provide the instance segmentation dataset preparation method used in the paper.

#### NWPU VHR-10 Dataset

- Image and Instance label download address: [NWPU VHR-10 Dataset](https://pan.baidu.com/s/1g9yXyGcFCf26Qkq4gqi_KQ). Password: 2017.

#### SSDD Dataset

- Image and Instance label download address: [SSDD Dataset](https://pan.baidu.com/s/1fu3A2R9JIlQlOF6c7XcFdA). Password: 2017.


#### Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format:

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/NWPU
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017
└── val2017
```
Note: In the project folder, we provide a folder named `data`, which contains examples of the organization method of the above datasets.

### Other Datasets

If you want to use other datasets, you can refer to [MMDetection documentation](https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/dataset_prepare.html) to prepare the datasets.

</details>

## Checkpoint Preparation
You can get the checkpoint of SAM on the [SAM](https://github.com/facebookresearch/segment-anything) project.
Then, use the `sep_module.py` to divide the model into three modules: image encoder, prompt_encoder and mask_decoder.
```shell
python segment_anything/sep_module.py
```
Note: Please change the path to the checkpoint of SAM in the `sep_module.py`
## Model Training

### CSNet Model

#### Config File and Main Parameter Parsing

We provide the configuration files of the CSNet models used in the paper, which can be found in the `configs/csnet` folder. The Config file is completely consistent with the API interface and usage method of MMDetection. 


#### Single Card Training

```shell
python tools/train.py configs/csnet/xxx.py  # xxx.py is the configuration file you want to use
```

#### Multi-card Training

```shell
sh ./tools/dist_train.sh configs/csnet/xxx.py ${GPU_NUM}  # xxx.py is the configuration file you want to use, GPU_NUM is the number of GPUs used
```

### Other Instance Segmentation Models

<details>

If you want to use other instance segmentation models, you can refer to [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main) to train the models, or you can put their Config files in the `configs` folder of this project, and then train them according to the above methods.

</details>

## Model Testing

#### Single Card Testing:

```shell
python tools/test.py configs/csnet/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```

#### Multi-card Testing:

```shell
sh ./tools/dist_test.sh configs/csnet/xxx.py ${CHECKPOINT_FILE} ${GPU_NUM}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```


## Image Prediction

#### Single Image Prediction:

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/csnet/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE is the image file you want to predict, xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```

#### Multi-image Prediction:

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs/csnet/xxx.py --weights ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_DIR is the image folder you want to predict, xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```


## Acknowledgement

This project is developed based on the [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main) project. Thanks to the developers of the MMDetection project.


## Contact

If you have any other questions❓, please contact us in time 👬

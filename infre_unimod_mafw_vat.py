import torch
import argparse
from torch.utils.data import DataLoader
import torchvision as tv
from typing import Type
from typing import Union
from typing import Optional
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from model_unimod_mafw_vat import JMPF
import numpy as np
import io
import os
import glob
import json
import sys
import time
import math
import signal
import argparse
import numpy as np
from collections import defaultdict
import contextlib
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
from copy import deepcopy
from data_prepro_mafw_test_ori import data_prepro_t

devices=[4]

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
USE_CUDA = torch.cuda.is_available()
device_ids_parallel = [4]
device = torch.device("cuda:{}".format(device_ids_parallel[0]) if USE_CUDA else "cpu")

BATCH_TRAIN = 8
BATCH_TEST = 8
WORKERS_TRAIN = 4
WORKERS_TEST = 4
EPOCHS = 100
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')
import torch
# 设置矩阵乘法的精度为中等或高等
torch.set_float32_matmul_precision('medium')

num_gpus = torch.cuda.device_count()

model = JMPF()

test_dataset = data_prepro_t(root = '/home/Datasets/DB_ER')
len_dataset = test_dataset.__len__()
print('len_dataset:',len_dataset)

# train_len = int (train_ratio * len_dataset) # 训练集长度
# valid_len = math.ceil (valid_ratio * len_dataset) # 验证集长度
# test_len = len_dataset - train_len - valid_len # 测试集长度

# train_dataset, val_dataset, test_dataset = random_split (dataset, lengths=[train_len, valid_len, test_len], generator=torch.Generator().manual_seed (0))


test_loader = DataLoader(test_dataset, batch_size=BATCH_TEST, shuffle=False, num_workers=4) 


checkpoint = torch.load("/home/et23-maixj/mxj/JMPF_pooling/checkpoint/unimod_mafw_vat.ckpt", map_location = device) # 这里假设您的checkpoint文件名是best-checkpoint.ckpt，您可以根据您的实际文件名来修改

# 将模型的参数加载到您的模型中
model.load_state_dict(checkpoint["state_dict"],strict=True)

# 设置模型为评估模式，这样可以关闭一些影响推理结果的功能，比如dropout和batch normalization
model.eval()

trainer = Trainer(
    accelerator="gpu", 
    devices=devices,
)

trainer.test(model,test_loader)



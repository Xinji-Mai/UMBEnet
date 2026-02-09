import torch
import argparse
from torch.utils.data import DataLoader
import torchvision as tv #noqa
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

# from model_eee import JMPF  #485
from model_unimod_mafw_vat2va import JMPF #
from data_prepro_mafw_ori import data_prepro
from data_prepro_mafw_test_ori import data_prepro_t

from pytorch_lightning.loggers import TensorBoardLogger
filename = "unimod_mafw_vat2va"

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
devices = [4]

ckp_path = '/home/et23-maixj/mxj/JMPF_pooling/checkpointChange/unimod_mafw_vat.ckpt'
# ckp_path = "/home/et23-maixj/mxj/JMPF_pooling/checkpoint/2scream_vat_7_fuse_ab1.ckpt"

# 加载原始checkpoint
checkpoint = torch.load(ckp_path, map_location='cpu')

ckp_path = None

BATCH_TRAIN = 8
BATCH_TEST = 8
WORKERS_TRAIN = 8
WORKERS_TEST = 8
EPOCHS = 30
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')
# 设置矩阵乘法的精度为中等或高等
torch.set_float32_matmul_precision('medium')

model = JMPF()

# 将模型的参数加载到您的模型中
model.load_state_dict(checkpoint["state_dict"],strict=True)

dataset = data_prepro(root='/home/Datasets/DB_ER')
dataset_t = data_prepro_t(root='/home/Datasets/DB_ER')
len_dataset = dataset.__len__()
print('len_dataset:', len_dataset)

# train_len = int(train_ratio * len_dataset)  # 训练集长度
# valid_len = math.ceil(valid_ratio * len_dataset)  # 验证集长度
# test_len = len_dataset - train_len - valid_len  # 测试集长度

# train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[
#                                                     train_len, valid_len, test_len], generator=torch.Generator().manual_seed(0))

# train_dataset, val_dataset = random_split(dataset, lengths=[
#                                                     train_len, valid_len], generator=torch.Generator().manual_seed(0))


batch_train_size = BATCH_TRAIN  # 定义批次大小
batch_test_size = BATCH_TEST  # 定义批次大小
shuffle = True  # 定义是否打乱数据
# train_loader = DataLoader(
# train_dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=2)
# val_loader = DataLoader(
# val_dataset, batch_size=batch_train_size, shuffle=False, num_workers=2)
# test_loader = DataLoader(
# test_dataset, batch_size=batch_test_size, shuffle=False, num_workers=2)
# 创建DataLoader实例，传入dataset实例和其他参数

train_loader = DataLoader(
dataset, batch_size=batch_train_size, shuffle=shuffle, num_workers=2)
val_loader = DataLoader(
dataset_t, batch_size=batch_train_size, shuffle=False, num_workers=2)

logger = TensorBoardLogger('/home/et23-maixj/mxj/JMPF/runs', name='JMPF')

checkpoint_callback = ModelCheckpoint(
dirpath="/home/et23-maixj/mxj/JMPF_pooling/checkpoint",
filename=filename,
save_top_k=1,  # 保存最好的1个checkpoint
verbose=True,
monitor="val_loss",  # 根据验证集损失来判断最好的checkpoint
mode="min"  # 最小化验证集损失
)

trainer = Trainer(
max_epochs=EPOCHS,
accelerator="gpu",
devices=devices,
callbacks=[checkpoint_callback],  # 是否启用checkpoint回调
strategy='ddp_find_unused_parameters_true',
logger=logger
)

trainer.fit(model, train_dataloaders=train_loader,
        val_dataloaders=val_loader, ckpt_path=ckp_path)
trainer.test(model,val_loader)

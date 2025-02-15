import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import random
import cv2
import torchvision.transforms as transforms
from PIL import Image
import natsort
import numpy as np
from utils import transforms as u_transforms
import librosa
from typing import Optional
from typing import Tuple
class data_prepro(Dataset):
    def __init__(self, 
                 root='/home/Datasets/DB_ER',
                 ):
        self.dataset_name = "MAFW"
        super(data_prepro, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.f_image_dir = os.path.join(self.dataset_dir, "face")
        self.o_image_dir = os.path.join(self.dataset_dir, "frames")
        # self.p_image_dir = os.path.join(self.dataset_dir, "preprocess/person")
        self.audio_dir = os.path.join(self.dataset_dir, "caudios")
        self.items_to_load = {}
        maskxlsx = pd.read_csv(os.path.join(self.dataset_dir,'preprocess/train_random.csv'))
        allcsv = pd.read_excel(os.path.join(self.dataset_dir,'Labels/single-set.xlsx'),dtype={'name': str})
        text = pd.read_excel(os.path.join(self.dataset_dir,'Labels/descriptive_text.xlsx'))
        self.meta = []
        meta = pd.merge(maskxlsx,allcsv,on='name')
        meta = pd.merge(meta,text,on='name')

        self.sample_rate = 22050
        self.audio_data = {}
        self.load_audio_data()

        
        classnames = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'neutral',5:'sadness',6:'surprise',
                      7:'contempt',8:'anxiety',9:'helplessness',10:'disappointment'}

        for idx, row in meta.iterrows():
            if row['mask'] == 1:
                name = row['name'].split('.mp4')[0]
                cap = row['e_cap']
                label = row['single_label'] - 1
                # print(name,cap,label)
                self.meta.append([name,cap,label])
            else:
                pass

            
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.transforms_audio = transforms.Compose([
            u_transforms.ToTensor1D(),
            u_transforms.RandomPadding(out_len = 110250, train = False),
            u_transforms.RandomCrop(out_len = 110250, train = False)
        ])

    def __len__(self):
        return len(self.meta)
    
    @staticmethod
    def _load_worker(ident:str, idx: int, filename: str, sample_rate: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
        wav, sample_rate = librosa.load(filename, sr=sample_rate, mono=True)
        # 计算目标长度，假设为5秒
        target_length = 3 * sample_rate
        # 使用trim函数去除静音部分，并获取索引范围
        wav, index = librosa.effects.trim(wav, top_db=20)
        # 如果去除静音后的长度小于目标长度，则在末尾填充零
        if len(wav) < target_length:
            wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
        # 如果去除静音后的长度大于目标长度，则从中间截取一段
        elif len(wav) > target_length:
            # 计算中间位置
            mid = (index[0] + index[1]) // 2
            # 计算起始位置和结束位置
            start = mid - target_length // 2
            end = mid + target_length // 2
            # 截取音频信号
            wav = wav[start:end]

        if wav.ndim == 1:
            wav = wav[:, np.newaxis]

        wav = wav.T * 32768.0

        return ident, idx, sample_rate, wav.astype(np.float32)
    
    def load_audio_data(self):
        
        audio_lists = os.listdir(self.audio_dir)
        tidx = 0

        for au in audio_lists:
            ident = self.audio_dir + '/' + au.split('.mp3')[0]
            au_path = os.path.join(self.audio_dir, au)
            temp = ident, tidx, au_path, self.sample_rate
            self.items_to_load[ident] = temp
            tidx = tidx + 1
        
    def get_all_audio(self, audio_dir):

        try:
            # 尝试从字典中获取值
            ident, idx, au_path, sample_rate = self.items_to_load[audio_dir]
            ident, idx, sample_rate, wav = self._load_worker(ident=ident, idx=idx, filename=au_path, sample_rate=sample_rate)
            try:
                wav = self.transforms_audio(wav)
                return wav
            except IndexError:
                print(ident)
                return torch.randn(1,110250)
        except KeyError:
            # 如果遇到KeyError，执行备选方案
            # print(f"Warning: Key {audio_dir} not found in items_to_load.")
            return torch.randn(1,110250)
 
        

    def __getitem__(self, idx):
        imgs_dir, cap, label = self.meta[idx]
        audio = self.get_all_audio(os.path.join(self.audio_dir,imgs_dir))
        f_frames, valid_list = self.get_all_video_frame(os.path.join(self.f_image_dir,imgs_dir))
        o_frames, valid_list = self.get_all_video_frame_o(os.path.join(self.o_image_dir,imgs_dir))
        # p_frames, valid_list = self.get_all_video_frame(os.path.join(self.p_image_dir,imgs_dir))
        return f_frames, o_frames, cap, label, audio, valid_list

    def get_all_video_frame(self, orignal_path):
        video_x = list()
        img_lists = os.listdir(orignal_path)
        img_lists = natsort.natsorted(img_lists)

        img_lists = os.listdir(orignal_path)

        img_lists = [f for f in img_lists if f.endswith(".jpg")]
        img_count = len(img_lists)

        # print('img_count',img_count)
        AllFrames = 16
        valid_list = torch.ones(AllFrames)
        # pred = self.get_pred_img(img_count,orignal_path,img_lists)
        if(img_count < AllFrames):
            # print('###################')
            # print(img_count)
            img_first = Image.new("RGB", (0, 0))
            for i in range(img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)

                img_first = self.transform(img_first)
                video_x.append(img_first)
            addFrameNumber = AllFrames - img_count
            for i in range(addFrameNumber):
                video_x.append(img_first)
                valid_list[len(video_x)-1] = 0
            video_x = torch.stack(video_x, dim=0)
        else:
            for i in range(img_count - AllFrames, img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)
                # img_first.save('/home/et23-maixj/mxj/crossaiclip/cache/' + orignal_path.split('/home/et23-maixj/mxj/DFER_Datasets/MAFW/preprocess')[0] + str(i) + '.jpg')
                img_first = self.transform(img_first)
                video_x.append(img_first)
                #3 224 224
            video_x = torch.stack(video_x, dim=0)
        # 3 16 224 224
        return video_x,valid_list
    
    def get_all_video_frame_o(self, orignal_path):
        video_x = list()
        img_lists = os.listdir(orignal_path)
        img_lists = natsort.natsorted(img_lists)

        img_lists = os.listdir(orignal_path)

        img_lists = [f for f in img_lists if f.endswith(".png")]
        img_count = len(img_lists)

        # print('img_count',img_count)
        AllFrames = 16
        valid_list = torch.ones(AllFrames)
        # pred = self.get_pred_img(img_count,orignal_path,img_lists)
        if(img_count < AllFrames):
            # print('###################')
            # print(img_count)
            img_first = Image.new("RGB", (0, 0))
            for i in range(img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)

                img_first = self.transform(img_first)
                video_x.append(img_first)
            addFrameNumber = AllFrames - img_count
            for i in range(addFrameNumber):
                video_x.append(img_first)
                valid_list[len(video_x)-1] = 0
            video_x = torch.stack(video_x, dim=0)
        else:
            for i in range(img_count - AllFrames, img_count):
                path_first_image = os.path.join(orignal_path, img_lists[i])
                img = cv2.imread(path_first_image)
                get_frame_same_size = np.zeros((112, 168, 3))
                height_scale = 110 / img.shape[0]
                width_scale = 160 / img.shape[1]
                if (height_scale < 1 or width_scale < 1):
                    scale_min = min(height_scale, width_scale)
                    img = cv2.resize(img, None, fx=scale_min, fy=scale_min, interpolation=cv2.INTER_CUBIC)
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img
                else:
                    get_frame_same_size[: img.shape[0], : img.shape[1], : img.shape[2]] = img

                get_frame_same_size = get_frame_same_size.astype(np.uint8)
                get_frame_same_size = cv2.cvtColor(get_frame_same_size, cv2.COLOR_BGR2RGB)
                img_first = Image.fromarray(get_frame_same_size)
                # img_first.save('/home/et23-maixj/mxj/crossaiclip/cache/' + orignal_path.split('/home/et23-maixj/mxj/DFER_Datasets/MAFW/preprocess')[0] + str(i) + '.jpg')
                img_first = self.transform(img_first)
                video_x.append(img_first)
                #3 224 224
            video_x = torch.stack(video_x, dim=0)
        # 3 16 224 224
        return video_x,valid_list
    
    def get_label_to_cate(self):
            return self.cls_num_list
    


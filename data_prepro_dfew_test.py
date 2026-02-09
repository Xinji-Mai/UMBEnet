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
# from torchvision.utils import save_image
class data_prepro_t(Dataset):
    def __init__(self, 
                 root='/home/et23-maixj/mxj/DFER_Datasets',
                 ):
        # root = '/home/wangy/DFER_Datasets'
        self.dataset_name = "DFEW/DFEW"
        super(data_prepro_t, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.f_image_dir = os.path.join(self.dataset_dir, "Image")
        self.o_image_dir = os.path.join(self.dataset_dir, "Image")
        # self.p_image_dir = os.path.join(self.dataset_dir, "preprocess/person")
        
        maskxlsx1 = pd.read_csv(os.path.join(self.dataset_dir,'EmoLabel/test_1.csv'))
        # maskxlsx2 = pd.read_csv(os.path.join(self.dataset_dir,'EmoLabel/test_2.csv'))
        # maskxlsx = pd.concat([maskxlsx1,maskxlsx2],axis=1)
        # allcsv = pd.read_excel(os.path.join(self.dataset_dir,'Labels/single-set.xlsx'),dtype={'name': str})
        # text = pd.read_excel(os.path.join(self.dataset_dir,'Labels/descriptive_text.xlsx'))

        # meta = pd.merge(maskxlsx,allcsv,on='name')
        # meta = pd.merge(meta,text,on='name')

        self.meta = []
        classnames = {"Angry":0, "Disgust":1, "Fear":2, "Happy":3, "Neutral":4, "Sad":5, "Surprise":6}
        classnames_change_39k = {4:0,6:1,7:2,1:3,3:4,2:5,5:6,}
        
        for idx, row in maskxlsx1.iterrows():
            path = row[0]
            # label = row[1] - 1
            label = classnames_change_39k[row[1]]
            temp = []
            temp.append(path)
            temp.append(label)
            self.meta.append(temp)

        # for idx, row in maskxlsx2.iterrows():
        #     path = row[0]
        #     label = row[1] - 1
        #     temp = []
        #     temp.append(path)
        #     temp.append(int(label))
        #     self.meta.append(temp)
            
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) #changemust
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]

        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),
        #     transforms.Resize((112, 112)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        imgs_dir = self.meta[idx][0]
        label = self.meta[idx][1]
        f_frames, valid_list = self.get_all_video_frame(os.path.join(self.f_image_dir,str(imgs_dir).zfill(5)))
        # f_frames = torch.zeros(16,3,224,224)
        # o_frames, valid_list = self.get_all_video_frame(os.path.join(self.o_image_dir,imgs_dir))
        return f_frames,f_frames, label, valid_list
        # return f_frames, label

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
                # save_image(img_first, '/home/et23-maixj/mxj/SIRV_baseline/icache/'+str(i)+'.jpg')
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
    


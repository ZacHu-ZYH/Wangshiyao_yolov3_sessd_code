import cv2
import glob
import os
import numpy as np
import torch
from MultiBoxLoss import point_form,match
from model import PriorBox
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transform
class Dateset(Dataset):
    def __init__(self,argument=True,img_size=448,batch_size=12,train_type='train'):
        self.img_size=img_size
        self.argument=argument
        self.batch_size=batch_size
        self.img_size=img_size
        self.min_size=self.img_size-3*32
        self.max_size=self.img_size+3*32
        self.img_list=os.listdir('./dataset/{}/img/'.format(train_type))
        self.img_list.sort(key=lambda x: int(x[:-4]))
        self.img_path_list = []
        self.img_list_full_path(train_type)
        self.label_list = []
        self.label_bbox_list(train_type)
    def label_bbox_list(self,train_type):
        with open('./dataset/{}/{}_txt/{}_label.txt'.format(train_type,train_type,train_type),'r',encoding='utf-8') as f:
            label_=f.readlines()
            for aa in label_:
              self.label_list.append(aa.strip('\n').split(' ', 1)[1])
    def img_list_full_path(self,train_type):
        for index in range(len(self.img_list)):
            self.img_path_list.append('./dataset/{}/img/'.format(train_type)+self.img_list[index])
    def __getitem__(self, item):
        label_float_list=[]
        label=self.label_list[item]
        img_path=self.img_path_list[item]
        for label_float in label.split(' '):
            label_float_list.append(float(label_float))
        label=np.array(label_float_list)
        label=label.reshape(-1,5)
        boxes=torch.from_numpy(label)
        targets=torch.zeros((len(boxes),5))
        targets[:,:]=boxes
        img = transform.ToTensor()(Image.open(img_path).convert('RGB'))
        return img,targets
    def __len__(self):
        return min(len(self.img_path_list),len(self.label_list))
    def collate_fn(self,batch):
        imgs,targets=list(zip(*batch))
        targets=[boxes for boxes in targets if boxes is not None]
        for i,boxes in enumerate(targets):
           for j,box in enumerate(boxes):
               box[0]+=1
        imgs=torch.stack([ img for img in imgs])
        return imgs,targets
    def horistontal_flip(self,images,targets):
        images=torch.flip(images,[1])
        targets[:,2]=1-targets[:,2]
        return  images,targets
    def vertical_flip(self,images,targets):
        images=torch.flip(images,[2])
        targets[:,1]=1-targets[:,1]
        return images,targets

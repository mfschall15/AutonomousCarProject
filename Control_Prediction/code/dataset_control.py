import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import csv

def get_data_gt(data_path,gt_path):
    data_list = []
    gt_list = []
    for dire in os.listdir(data_path):
        if dire == '.DS_Store':
            continue
        path1 = os.path.join(data_path,dire)
        data_list+=sorted(os.path.join(path1,img) for img in os.listdir(path1))
        
    csv_gt = csv.reader(open(gt_path))
    i = 0
    for row in csv_gt:
        if (i==0):
            i+=1
            continue
        gt_list.append(row)
        
    return data_list,gt_list

class Lane_DBList(Dataset):
    def __init__(self,data_path,gt_path):
        self.data_list, self.gt_list = get_data_gt(data_path,gt_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # input and target label 
        in_name = self.data_list[index]
        img_names = np.array(self.gt_list)[:,0]
        gt_index = np.argwhere(img_names==str(in_name.split('/')[-1]))
       
        # process the images
        transf_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        in_image = Image.open(in_name).convert("RGB")
        width = in_image.width
        height = in_image.height
        in_image = transf_img(in_image)
        gt_label = self.gt_list[gt_index[0,0]][1:]
        x = float(gt_label[0])
        y = float(gt_label[1])
#         x = 2.0 * (float(gt_label[0]) / width - 0.5) # -1 left, +1 right
#         y = -2.0 * (float(gt_label[1]) / height - 0.8) # -1 top, +1 bottom
        #gt_label = torch.tensor([x,y])
        
        return in_image, torch.tensor([x,y])
    
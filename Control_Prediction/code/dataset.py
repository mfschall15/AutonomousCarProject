import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image

def generate_txt(img_path,gt_path,txt_file):
    with open(txt_file,'w') as f:
        for dir in os.listdir(img_path):
            if dir == '.DS_Store':
                continue
            path = osp.join(img_path,dir)
            f.write(path+'/'+dir+'_9.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_10.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_11.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_12.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_13.jpg')
            f.write('\n')
            f.write(path+'/'+dir+'_5.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_7.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_9.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_11.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write('\n')
            f.write(path+'/'+dir+'_1.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_4.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_7.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_10.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write('\n')
            f.write(path+'/'+dir+'_16.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_17.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_18.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_19.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_20.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_20.jpg')
            f.write(' ')
            f.write('\n')
            f.write(path+'/'+dir+'_12.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_14.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_16.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_18.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_20.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_20.jpg')
            f.write('\n')
            f.write(path+'/'+dir+'_8.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_11.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_14.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_17.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_20.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_20.jpg')
            f.write('\n')
    f.close()

def generate_test_txt(img_path,gt_path,txt_file):
    with open(txt_file,'w') as f:
        for dir in os.listdir(img_path):
            if dir == '.DS_Store':
                continue
            path = osp.join(img_path,dir)
            f.write(path+'/'+dir+'_9.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_10.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_11.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_12.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_13.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_13.jpg')
            f.write('\n')
            f.write(path+'/'+dir+'_16.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_17.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_18.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_19.jpg')
            f.write(' ')
            f.write(path+'/'+dir+'_20.jpg')
            f.write(' ')
            f.write(gt_path+'/'+dir+'/'+dir+'_20.jpg')
            f.write('\n')
    f.close()

def get_data_list(file):
    data_list = []
    with open(file,'r') as f:
        for line in f.readlines():
            data_list.append(line.split())
    f.close()
    return data_list

class Road_DBList(Dataset):
    def __init__(self,file):
        self.data_list = get_data_list(file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # input and target images
        in_names = self.data_list[index][:5]
        gt_name = self.data_list[index][5]
    
        # process the images
        transf_img = transforms.Compose([transforms.ToTensor()])
        in_images = []
        for name in in_names:
            in_image = Image.open(name).convert('RGB')
            in_image = torch.unsqueeze(transf_img(in_image),dim=0)
            in_images.append(in_image)
        in_images = torch.cat(in_images,0)
        gt_label = Image.open(gt_name)
        gt_label = torch.squeeze(transf_img(gt_label))
        
        #print(gt_label)
        '''gt1 = gt_label.clone()
        gt1 = torch.round(gt1).type(torch.LongTensor).to(torch.device('cuda'))
        gt1 = torch.squeeze(gt1).cpu().numpy()
        gt1 *= 255
        gt1 = gt1.astype(np.uint8)
        #print(gt1)
        gt1 = Image.fromarray(gt1)
        print("Saving")
        gt1.save('./test.jpg')'''

        return in_images, gt_label
    

#print(get_data_list('./Robust-Lane-Detection/LaneDetectionCode/data/test_index_sim.txt'))
generate_txt('./data/224_by_224/train/img','./data/224_224_binary', './data/224_train.txt')
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
from PIL import Image
import time
import os 

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def save_images(img,pred,save_pth,dire,j):
    #Saving prediction output to results folder
    img = img[-1].transpose([1,2,0])*255
    img = img.astype(np.uint8)
    
    pred *= 255
    pred = pred.astype(np.uint8)
    
    indices = np.argwhere(pred>200)
    img[indices[:,0],indices[:,1],:] = [255,0,0]
    
    img = Image.fromarray(img).convert('RGB')
    pred = Image.fromarray(pred)
    
    if not os.path.exists(save_pth+dire):
        os.makedirs(save_pth+dire)
    pred.save(save_pth+dire+'/'+dire+'_'+str(j)+'.jpg')
    #img.save(save_pth+str(k)+'_img.jpg')
    
def test(model,
        test_seq_path,
        dire,
        weights_path,
        save_pth = "./results/sim_realtime/"):
    #Test function to directly take in images
    im_tranforms = transforms.Compose([transforms.ToTensor()])
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    data = []
    k = 1
    #path = './data/sim_dataset_full/test/img/21/21_'
    path = test_seq_path
    im1 = torch.unsqueeze(im_tranforms(Image.open(path+'1.jpg')), dim=0)
    data.append(im1)
    im2 = torch.unsqueeze(im_tranforms(Image.open(path+'2.jpg')), dim=0)
    data.append(im2)
    im3 = torch.unsqueeze(im_tranforms(Image.open(path+'3.jpg')), dim=0)
    data.append(im3)
    im4 = torch.unsqueeze(im_tranforms(Image.open(path+'4.jpg')), dim=0)
    data.append(im4)
    tic = time.time()
    
    preds = []
    for j in range(5,21):
        tic1 = time.time()
        print(j)
        im5 = torch.unsqueeze(im_tranforms(Image.open(path+str(j)+'.jpg')), dim=0)
        data.append(im5)
        sample = torch.cat(data[j-5:j],0)
        sample = sample.view(1,sample.size(0),sample.size(1),sample.size(2),sample.size(3)).to(device)
        with torch.no_grad():
            score = model(sample)
        pred = torch.squeeze(score.max(1,keepdim=True)[1]).cpu().numpy()
        preds.append(pred)
        if save_pth is not None:
            save_images(torch.squeeze(sample).cpu().numpy(),pred,save_pth,dire,j)
        k+=1
        toc1 = time.time()
        print("Time Taken for 1 image: ",toc1-tic1)
    toc = time.time()
    print("Time Taken for 1 sequence: ",toc-tic)
    return preds
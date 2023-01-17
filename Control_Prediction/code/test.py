import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from PIL import Image
import time

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def save_images(img,pred,gt,save_pth,i):
    #Saving prediction output to results folder
    img = img[-1].transpose([1,2,0])*255
    img = img.astype(np.uint8)

    gt *= 255
    gt = gt.astype(np.uint8)
    
    pred *= 255
    pred = pred.astype(np.uint8)
    
    indices = np.argwhere(pred>200)
    img[indices[:,0],indices[:,1],:] = [255,0,0]
    
    img = Image.fromarray(img).convert('RGB')
    gt = Image.fromarray(gt)
    pred = Image.fromarray(pred)
    
    pred.save(save_pth+str(i)+'_pred.jpg')
    gt.save(save_pth+str(i)+'_gt.jpg')
    img.save(save_pth+str(i)+'_img.jpg')
    
def test(test_loader,
        model,
        weights_path,
        save_pth='./results/'):
    #Test function
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.02,1.02])).to(device)
    test_loss = 0
    test_acc = 0
    precision = 0
    recall = 0
    F1_score = 0
    i = 0
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    for (data, target) in test_loader:
        print(i)
        data, target = data.to(device), torch.round(target).type(torch.LongTensor).to(device)
        with torch.no_grad():
            score = model(data)
        loss = loss_func(score, target)
        test_loss += loss.item()
        pred = torch.squeeze(score.max(1,keepdim=True)[1]).cpu().numpy()
        gt = torch.squeeze(target).cpu().numpy()
        test_acc += ((pred==gt).sum()/(pred.shape[0]*pred.shape[1]))
        precision_new = ((pred*gt).sum()/np.count_nonzero(pred))
        recall_new = ((pred*gt).sum()/np.count_nonzero(gt))
        precision += precision_new
        recall += recall_new
        F1_score += (2*precision_new*recall_new/(precision_new+recall_new))
        #print(precision_new,recall_new,F1_score)
        save_images(torch.squeeze(data).cpu().numpy(),pred,gt,save_pth,i)
        i+=1
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    precision /= len(test_loader)
    recall /= len(test_loader)
    F1_score /= len(test_loader)
    
    print("Test Accuracy: {:.4}, Test Loss: {:.4}".format(
        test_acc, test_loss
        ))
    print("Precision: {:.2}, Recall: {:.2}, F1_score: {:.2}".format(
        precision, recall, F1_score
        ))
    return test_loss, test_acc, precision, recall, F1_score
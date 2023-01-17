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
    
def test(test_loader,
        model,
        weights_path,
        ):
    #Test function
    loss_func = nn.MSELoss().to(device)
    test_loss = 0
    i = 0
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    for (data, target) in test_loader:
        print(i)
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            score = model(data)
        loss = loss_func(score, target)
        test_loss += loss.item()
        print(score,target)
        i+=1
    
    test_loss /= len(test_loader)
    
    print("Test Loss: {:.4}".format(test_loss))
   
    return test_loss
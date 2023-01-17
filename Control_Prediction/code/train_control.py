import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import time

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def Evaluate(
    val_loader,
    model,
    loss_func
):
    val_loss = 0
    
    model.eval()
    for (data, target) in val_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            score = model(data)
        loss = loss_func(score, target)
        val_loss += loss.item()
    
    val_loss /= len(val_loader)

    return val_loss

def Train(
    model,
    loss_func,
    optim,
    scheduler,
    epochs,
    train_loader,
    val_loader,
    weights_pth,
    save_pth
):
    
    if weights_pth is not None:
        pretrained_dict  = torch.load(weights_pth)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    val_loss = Evaluate(
        val_loader,
        model,
        loss_func
    )
    
    print("Init Model")
    print("Validation Loss: {:.4}".format(
        val_loss
    ))
    for i in range(epochs):
        tic = time.time()
        print("Epochs: {}".format(i))
        total_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to("cuda:0"), target.to("cuda:0")
            optim.zero_grad()

            score = model(data)
            loss = loss_func(score, target)
            loss_data = loss.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            optim.step()
            total_loss += loss.item()
            
        
        total_loss /= len(train_loader)
        model.eval()
        val_loss = Evaluate(
            val_loader,
            model,
            loss_func
        )
        scheduler.step(total_loss)
        print("Training Loss: {:.4}".format(
        total_loss
        ))
        print("Validation Loss: {:.4}".format(
        val_loss
        ))
        
        torch.save(model.state_dict(), save_pth+str(i)+'_'+str(round(val_loss,4))+'.pth')
        toc = time.time()
        print("Time Taken for Epoch:{} sec".format(toc-tic))

def Trainer(model, 
            train_loader,
            val_loader,
            num_epochs=30, 
            weights_pth=None,
            save_pth='Trained_Weights/'
            ):
    # define optimizer
    lr = 1e-3
    weight_decay = 1e-4
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # define learning rate schedule
    scheduler = ReduceLROnPlateau(
        optim, 'min', patience=3,
        min_lr=1e-10, verbose=True
    )
    
    # define loss function
    loss_func = nn.MSELoss()

    Train(model,loss_func,optim,scheduler,num_epochs,train_loader,val_loader,weights_pth,save_pth)

    return model


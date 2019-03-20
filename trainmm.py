import os
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as t
from dataset import net_input
from model.mobilefacenet import MobileFaceNet
from model.mobilenet import MobileNet
from torch.autograd import Variable
import torch.optim as optim
from dataset.net_input import HyperspectralDataset
from torchvision import transforms
from tensorboardX import SummaryWriter

net=MobileFaceNet()
#net=MobileNet()
writer=SummaryWriter()
use_gpu=torch.cuda.is_available

if use_gpu():
   net.cuda()
   #print('gpu is available')

#初始化参数
batchsize=64                       #批处理大小
lr=1e-4

def train(epochs):
    #载入训练集数据
    train_dataset=HyperspectralDataset('train')
    trainloader=DataLoader(dataset=train_dataset,batch_size=batchsize,shuffle=True,num_workers=4)     #num_worker多线程数目
    
    #目标函数与优化器
    criterion=nn.CrossEntropyLoss()
    #ptimizer=optim.SGD(net.parameters(),lr=lr)
    optimizer=optim.Adam(net.parameters(),lr=lr)
    #optimizer=optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    
    #开始训练    
    for epoch in range(epochs):
        running_loss=0.0
        train_correct=0
        train_total=0
        net.train()
        for i,(inputs,train_labels) in enumerate(trainloader):                     
            if use_gpu():
                inputs,labels=Variable(inputs.cuda()),Variable(train_labels.cuda())
            else:
                inputs,labels=Variable(inputs),Variable(train_labels) 
            #inputs,labels=Variable(inputs.float()),Variable(train_labels)
            
            optimizer.zero_grad()
            outputs=net(inputs)                                                   #网络输出
            
            _,train_predicted=torch.max(outputs.data,1)
            
            train_correct += int(torch.sum(train_predicted.eq(labels.data)))
            
            loss =criterion(outputs,labels)
           
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            train_total+=int(train_labels.size(0))
           
        loss_train=running_loss/train_total
        acc_train=100*train_correct/train_total
         
        #载入校验集数据
        valid_dataset=HyperspectralDataset('valid')
        validloader=DataLoader(dataset=valid_dataset,batch_size=batchsize,shuffle=True,num_workers=4)     #num_worker多线程数目
        #目标函数与优化器
        criterion=nn.CrossEntropyLoss()
        
        net.eval()                                                          #在测试模型的时候使用
        
        valid_loss = 0
        valid_correct=0
        valid_total=0
        
        for j,(validinputs,valid_labels) in enumerate(validloader):
            if use_gpu():
                inputs,labels=Variable(validinputs.cuda()),Variable(valid_labels.cuda())
            else:
                inputs,labels=Variable(validinputs),Variable(valid_labels) 
         
         
            #inputs,labels=Variable(validinputs.float()),Variable(valid_labels)
            
            outputs=net(inputs)                                                   #网络输出
            
            _,valid_predicted=torch.max(outputs.data,1)
           
            valid_correct += int(torch.sum(valid_predicted.eq(labels.data)))
                      
            loss =criterion(outputs,labels)
            
            valid_loss+=loss.item()
            valid_total+=int(valid_labels.size(0))

        loss_valid=valid_loss/valid_total
        acc_valid=100*valid_correct/valid_total
                          
        print(' %d epoch  train  loss: %.3f  train_acc: %.3f        valid  loss:%.3f  valid_acc:%.3f' %(epoch+1,loss_train,acc_train,loss_valid,acc_valid))    

        #画图
        writer.add_scalars('mobilefaceaccuracy', {'train': acc_train,  'valid': acc_valid},  epoch)
        writer.add_scalars('loss',  {'train': loss_train, 'valid': loss_valid}, epoch)     

        #保存网络模型
        name = '/home/siminzhu/mmface_recognition/savepoints/mobilefacenet_'+str(epoch+1)+'.pkl'
        torch.save(net,name)   
        

train(100)

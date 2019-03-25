import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as t
from dataset import net_input
from model.mobilefacenet_arc import MobileFacenet
from model.mobilefacenet_arc import ArcMarginProduct
from model.mobilenet import MobileNet
from torch.autograd import Variable
import torch.optim as optim
from dataset.net_input import HyperspectralDataset
from torchvision import transforms
from tensorboardX import SummaryWriter
import time

net=MobileFacenet()
writer=SummaryWriter('log')
use_gpu=torch.cuda.is_available
ArcMargin = ArcMarginProduct(128,33)

if use_gpu():
   net.cuda()
   ArcMargin = ArcMargin.cuda()
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
    optimizer=optim.Adam(net.parameters(),lr=lr)
        
    #开始训练    
    for epoch in range(epochs):
        running_loss=0.0
        train_correct=0
        train_total=0
        since = time.time()
        net.train()
        for i,(inputs,train_labels) in enumerate(trainloader):                     
            if use_gpu():
                inputs,labels=Variable(inputs.cuda()),Variable(train_labels.cuda())
                inputs=inputs.contiguous()
            else:
                inputs,labels=Variable(inputs),Variable(train_labels) 
            optimizer.zero_grad()
            #outputs=net(inputs)                                                   #网络输出
            raw_logits = net(inputs)
            outputs = ArcMargin(raw_logits, labels)
            _,train_predicted=torch.max(outputs.data,1)
            
            train_correct += int(torch.sum(train_predicted.eq(labels.data)))
            
            loss =criterion(outputs,labels)
           
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            train_total+=int(train_labels.size(0))
           
        loss_train=100*running_loss/train_total
        acc_train=100*train_correct/train_total
         
        #载入校验集数据
        valid_dataset=HyperspectralDataset('valid')
        validloader=DataLoader(dataset=valid_dataset,batch_size=batchsize,shuffle=True,num_workers=8)     #num_worker多线程数目
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
            #outputs=net(inputs)  
            raw_logits = net(inputs)
            outputs = ArcMargin(raw_logits, labels)
                       
            _,valid_predicted=torch.max(outputs.data,1)
           
            valid_correct += int(torch.sum(valid_predicted.eq(labels.data)))
                      
            loss =criterion(outputs,labels)
            
            valid_loss+=loss.item()
            valid_total+=int(valid_labels.size(0))

        loss_valid=100*valid_loss/valid_total
        acc_valid=100*valid_correct/valid_total
        
        time_elapsed = time.time() - since
        print(' %d epoch  train  loss: %.3f  train_acc: %.3f    valid  loss:%.3f  valid_acc:%.3f   time:%.0fm %.0fs' %(epoch+1,loss_train,acc_train,loss_valid,acc_valid,time_elapsed // 60,time_elapsed % 60))    

        #画图
        writer.add_scalars('mobilefaceaccuracy', {'train': acc_train,  'valid': acc_valid},  epoch)
        writer.add_scalars('loss',  {'train': loss_train, 'valid': loss_valid}, epoch)     

        #保存网络模型
#         name = '/home/siminzhu/mmface_recognition/savepoints/mobilefacenet_'+str(epoch+1)+'.pkl'
#         torch.save(net,name)   
        

train(200)
torch.save(net.state_dict(),'3-25Mobilefacenet1.pkl')

def test(epochs):
    #载入测试集数据
    test_dataset=HyperspectralDataset('test')
    testloader=DataLoader(dataset=test_dataset,batch_size=batchsize,shuffle=True,num_workers=8)     #num_worker多线程数目
    
    #损失函数
    criterion=nn.CrossEntropyLoss()
        
    net.eval()                                                          #在测试模型的时候使用

    for epoch in range(epochs):
        start= time.time()
        test_loss = 0
        test_correct=0
        test_total=0

        for i,(inputs,train_labels) in enumerate(testloader):                     
            if use_gpu():
                inputs,labels=Variable(inputs.cuda()),Variable(train_labels.cuda())
                inputs=inputs.contiguous()
            else:
                inputs,labels=Variable(inputs),Variable(train_labels) 

            raw_logits = net(inputs)
            outputs = ArcMargin(raw_logits, labels)

            _,test_predicted=torch.max(outputs.data,1)
            test_correct += int(torch.sum(test_predicted.eq(labels.data)))

            loss =criterion(outputs,labels)
            test_loss+=loss.item()
            test_total+=int(test_labels.size(0))
        
        loss_test=test_loss/test_total
        acc_test=100*test_correct/test_total
        
        time_elapsed = time.time() -start
        print(' %d epoch  test  loss: %.3f  test_acc: %.3f   time:%.0fm %.0fs' %(epoch+1,loss_test,acc_test,time_elapsed // 60,time_elapsed % 60)) 

        #画图
        writer.add_scalars('mobilefacetestaccuracy', {'test': acc_test},  epoch)
        writer.add_scalars('loss',  {'test': loss_test}, epoch) 

test(200)


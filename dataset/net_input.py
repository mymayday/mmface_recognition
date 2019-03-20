import os                                           #os 模块提供了非常丰富的方法用来处理文件和目录
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

##数据加载     基本原理:使用Dataset封装数据集,再使用Dataloader实现数据并行加载
class HyperspectralDataset(Dataset):
    
    def __init__(self,mode,transforms=None):
        '''目标:根据.txt文件获取所有图片的地址 '''
        
        
        self.imgpath_list=[]
        
        if mode == "train":
            f = open("/home/siminzhu/mmface_recognition/dataset/train.txt", "r")
        elif mode == "valid":
            f = open("/home/siminzhu/mmface_recognition/dataset/valid.txt", "r")
        else:
            f = open("/home/siminzhu/mmface_recognition/dataset/test.txt", "r")
        contents=f.readlines()                                                #读取文档中的所有行
                
        for line in contents:
            ls=line.lstrip('/home/louishsu/Work/Workspace/ECUST2019/')           #lstrip() 删除 string 字符串开头的指定字符
            rs=ls.rstrip(ls.split('/')[-1])                            #rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
            xs=ls.lstrip(rs).split('_')[-1].rstrip('.bmp\n')
        
            newpath=rs.replace(rs,"/home/xianyi/ECUST2019_NPY_new/"+rs+xs+'.npy')
            
            self.imgpath_list.append(newpath)
            
        f.close()
            
    def __getitem__(self,index):
        '''返回一张图片的数据'''
        
        img=np.load(self.imgpath_list[index])
        data=np.resize(img,(64,64)).reshape((64,64,1))
        data_transform=transforms.Compose([
            transforms.ToTensor(),
        ]
        )                                                 
        data=data_transform(data)                                   #对加载的图像做归一化处理
        data=data.contiguous()
        label=int(self.imgpath_list[index].split('/',-1)[5])-1
        #print(data.size())
        #print(data,label)
        return data,label
    
    def __len__(self):
        '''返回数据集中所有图片的数目'''
        return len(self.imgpath_list)   

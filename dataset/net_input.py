import os                                           #os 模块提供了非常丰富的方法用来处理文件和目录
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from torchvision.transforms import ToTensor
from dataset.config import configer

notUsedSubjects = []
get_vol = lambda i: (i-1)//10+1


# #数据加载     基本原理:使用Dataset封装数据集,再使用Dataloader实现数据并行加载
#bymyself
# class HyperspectralDataset(Dataset):
    
#     def __init__(self,mode,transforms=None):
#         '''目标:根据.txt文件获取所有图片的地址 '''
       
        
#         self.imgpath_list=[]
        
        # if mode == "train":
        #     f = open(configer.traintxtpath, "r")
        # elif mode == "valid":
        #     f = open(configer.validtxtpath, "r")
        # else:
        #     f = open(configer.testtxtpath, "r")
#         contents=f.readlines()                                                #读取文档中的所有行
                
#         for line in contents:
#             ls=line.lstrip('/home/louishsu/Work/Workspace/ECUST2019/')           #lstrip() 删除 string 字符串开头的指定字符
#             rs=ls.rstrip(ls.split('/')[-1])                            #rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
#             xs=ls.lstrip(rs).split('_')[-1].rstrip('.bmp\n')
        
#             newpath=rs.replace(rs,"/home/xianyi/ECUST2019_NPY_new/"+rs+xs+'.npy')
            
#             self.imgpath_list.append(newpath)
            
#         f.close()
            
#     def __getitem__(self,index):
#         '''返回一张图片的数据'''
        
#         img=np.load(self.imgpath_list[index])
#         data=np.resize(img,(64,64)).reshape((64,64,1))
#         data_transform=transforms.Compose([
#             transforms.ToTensor(),
#         ]
#         )                                                 
#         data=data_transform(data)                                   #对加载的图像做归一化处理
#         data=data.contiguous()
#         label=int(self.imgpath_list[index].split('/',-1)[5])-1
#         #print(data.size())
#         #print(data,label)
#         return data,label
    
#     def __len__(self):
#         '''返回数据集中所有图片的数目'''
#         return len(self.imgpath_list)   

# def getDicts():
#     dicts = dict()
#     for vol in ["DATA%d" % _ for _ in range(1, 5)]:
#         txtfile = os.path.join(configer.datasetpath, vol, "detect.txt")
#         with open(txtfile, 'r') as f:
#             dicts[vol] = eval(f.read())
#     return dicts    

def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    label = int(path_split[idx+2])
    return label

# class HyperspectralDataset(Dataset):
#copy from xyb
#     labels = [i for i in range(1, 41) if (i not in notUsedSubjects)]
#     def __init__(self,mode,transforms=None):
#         '''目标:根据.txt文件获取所有图片的地址 '''
        
#         self.imgpath_list=[]
        
        # if mode == "train":
        #     f = open(configer.traintxtpath, "r")
        # elif mode == "valid":
        #     f = open(configer.validtxtpath, "r")
        # else:
        #     f = open(configer.testtxtpath, "r")
            

#         self.contents=f.readlines()                                                #读取文档中的所有行
#         self.facesize = tuple((64,64))
#         self.dicts = getDicts()
        
#     def __getitem__(self, index):
#         filename = self.contents[index].strip()
#         filename = os.path.join(configer.datasetpath, filename)
#         #filename = os.path.join("/datasets/ECUST2019_NPY", filename)
#         label = get_label_from_path(filename)

#         # get bbox
#         vol = "DATA%d" % get_vol(label)
#         imgname = filename[filename.find("DATA")+5:]
#         #print('imgname',imgname)
#         dirname = '/'.join(imgname.split('/')[:-1])
#         #print('dirname',dirname)
#         bbox = self.dicts[vol][dirname][1]
#         [x1, y1, x2, y2] = bbox

#         # load image array
#         image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
#         if self.facesize is not None:
#             image = cv2.resize(image, self.facesize[::-1])

#         image = image[:, :, np.newaxis]
#         image = ToTensor()(image)
#         label = self.labels.index(label)
#         return image, label
    
#     def __len__(self):
#         return len(self.contents)

        

#数据加载     基本原理:使用Dataset封装数据集,再使用Dataloader实现数据并行加载
#46通道输入
class HyperspectralDataset(Dataset):
    
    def __init__(self,mode,transforms=None):
        '''目标:根据.txt文件获取所有图片的地址 '''
               
        self.imgpath_list=[]                                                                                                                                
                                    
        if mode == "train":
            f = open(configer.traintxtpath, "r")
        elif mode == "valid":
            f = open(configer.validtxtpath, "r")
        else:
            f = open(configer.testtxtpath, "r")
        contents=f.readlines()                                                #读取文档中的所有行
                
        for line in contents:           
            xs=line.rstrip('\n')       
            newpath=xs.replace(xs,configer.datasetpath+'/'+xs+'.npy')
            #print(newpath)
            self.imgpath_list.append(newpath)
            
        f.close()
            
    def __getitem__(self,index):
        '''返回一张图片的数据'''
        
        img=np.load(self.imgpath_list[index])                #图片读进来(h,w,c)     照片尺寸hxw,通道数c
        h,w,c=img.shape
        data=[]
        for i in range(c):
            single=img[:,:,i]
            newdata=np.resize(img,(64,64))
            print(newdata)
            data.append(newdata)
        data=np.reshape((64,64,46))
        print(data.shape)
        data_transform=transforms.Compose([
            transforms.ToTensor(),
        ]
        )                                                 
        data=data_transform(data)                                   #对加载的图像做归一化
        label=int(self.imgpath_list[index].split('/',-1)[3])-1
        print(data.size())
        print(data,label)
        return data,label
    
    def __len__(self):
        '''返回数据集中所有图片的数目'''
        return len(self.imgpath_list)   


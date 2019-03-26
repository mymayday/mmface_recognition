import os                                           #os 模块提供了非常丰富的方法用来处理文件和目录
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from torchvision.transforms import ToTensor

##数据加载     基本原理:使用Dataset封装数据集,再使用Dataloader实现数据并行加载
# class HyperspectralDataset(Dataset):
    
#     def __init__(self,mode,transforms=None):
#         '''目标:根据.txt文件获取所有图片的地址 '''
        
        
#         self.imgpath_list=[]
        
#         if mode == "train":
#             f = open("/home/siminzhu/mmface_recognition/dataset/train.txt", "r")
#         elif mode == "valid":
#             f = open("/home/siminzhu/mmface_recognition/dataset/valid.txt", "r")
#         else:
#             f = open("/home/siminzhu/mmface_recognition/dataset/test.txt", "r")
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

def getDicts():
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 5)]:
        txtfile = os.path.join(configer.datapath, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
return dicts    

def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    label = int(path_split[idx+2])
return label

class HyperspectralDataset(Dataset):
    labels = [i for i in range(1, 41) if (i not in notUsedSubjects)]
    def __init__(self,mode,transforms=None):
        '''目标:根据.txt文件获取所有图片的地址 '''
        
        self.imgpath_list=[]
        
        if mode == "train":
            f = open("/home/siminzhu/mmface_recognition/dataset/new split(相邻波段问题)/split1/train.txt", "r")
        elif mode == "valid":
            f = open("/home/siminzhu/mmface_recognition/dataset/new split(相邻波段问题)/split1/valid.txt", "r")
        else:
            f = open("/home/siminzhu/mmface_recognition/dataset/new split(相邻波段问题)/split1/test.txt", "r")
        self.contents=f.readlines()                                                #读取文档中的所有行
        self.facesize = tuple(facesize)
        self.dicts = getDicts()
        
    def __getitem__(self, index):
        filename = self.contents[index].strip()
        filename = os.path.join("/datasets/ECUST2019", filename)
        label = get_label_from_path(filename)

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        [x1, y1, x2, y2] = bbox

        # load image array
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])

        image = image[:, :, np.newaxis]
        image = ToTensor()(image)
        label = self.labels.index(label)
        return image, label
    
    def __len__(self):
        return len(self.contents)

        

# class HyperECUST(Dataset):
#     labels = [i for i in range(1, 41) if (i not in notUsedSubjects)]

#     def __init__(self, splitmode, facesize=None, mode='train'):
#         """
#         Params:
#             facesize:   {tuple/list[H, W]}
#             mode:       {str} 'train', 'valid'
#         """
#         with open('./split_23chs/{}/{}.txt'.format(splitmode, mode), 'r') as f:
#             self.filenames = f.readlines()
#         self.facesize = tuple(facesize)
#         self.dicts = getDicts()

#     def __getitem__(self, index):
#         filename = self.filenames[index].strip()
#         filename = os.path.join(configer.datapath, filename)
#         label = get_label_from_path(filename)

#         # get bbox
#         vol = "DATA%d" % get_vol(label)
#         imgname = filename[filename.find("DATA")+5:]
#         dirname = '/'.join(imgname.split('/')[:-1])
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
#         return len(self.filenames)


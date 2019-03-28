from torch.cuda import is_available
from easydict import EasyDict                 #Access dict values as attributes (works recursively)

configer = EasyDict()

#configer.datasetpath="/datasets/ECUST2019"
configer.datasetpath="/datasets/ECUST2019_NPY"
configer.traintxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/train.txt"
configer.validtxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/valid.txt"
configer.testtxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/test.txt"
# configer.logspath   =
# configer.modelspath =


configer.facesize       = (64, 64)
configer.n_channels     = 23
configer.n_classes      = 40
configer.usedChannels   =[550]

configer.modelname = "MobilefaceNet"

configer.learningrate  = 1e-4
configer.batchsize     = 64
configer.n_epoch       = 200

configer.cuda = is_available()
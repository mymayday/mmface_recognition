from torch.cuda import is_available
from easydict import EasyDict                            #Access dict values as attributes (works recursively)

configer = EasyDict()

configer.datasetpath="/datasets/ECUST2019"
configer.traintxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/train_rgb.txt"
configer.validtxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/valid_rgb.txt"
configer.testtxtpath="/home/siminzhu/mmface_recognition/dataset/new split(46通道输入)/split/test_rgb.txt"

configer.trainmode      = 'RGB'
configer.facesize       = (64, 64)
configer.n_channels     = 3
configer.n_classes      = 40

if configer.trainmode == 'Multi':
    configer.usedChannels    = [550]
    configer.n_usedChannels = len(configer.usedChannels)
    # configer.modelname = "{}_{}_{}chs_{}sta_20nm".\
    #             format(configer.modelbase, configer.splitmode, configer.n_usedChannels, configer.usedChannels[0])
elif configer.trainmode == 'RGB':
    configer.usedRGBChannels = 'RGB'
    configer.n_usedChannels = 3
    # configer.modelname = '{}_{}_{}'.\
    #             format(configer.modelbase, configer.splitmode, configer.usedRGBChannels)

configer.modelname = "MobilefaceNet"

configer.learningrate  = 1e-4
configer.batchsize     = 64
configer.n_epoch       = 200

configer.cuda = is_available()
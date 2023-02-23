import os, sys
sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the executors
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './utils')
#sys.path.insert(0, './pLoss')
#sys.path.insert(0, './models')
import random, math
from glob import glob
from tqdm import tqdm
import logging
from statistics import median
# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# --------
from torch.utils.tensorboard import SummaryWriter
import torchio as tio
from torchio.data.io import read_image
import nibabel as nib

from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)

from utilities_new import MoCoDatasetRegressionUpdated, tensorboard_regression, getSSIM
# from pLoss.perceptual_loss import PerceptualLoss
# from models.ReconResNetV2 import ResNet

### ----------------------------------------------------- ###
# os.environ["CUDA_VISIBLE_DEVICES"] = 0
# cuda = 1
# non_deter = False
# seed = 1701
# torch.backends.cudnn.benchmark = non_deter 
# if not non_deter:
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

device = torch.device("cuda:2") # if torch.cuda.is_available() and cuda else "cpu")
log_path = './TBLogs'
trainID = 'TEST-RN101-Regression-TIO-RealityMot-Combined-NoContrastAug'
save_path = './Results'
tb_writer = SummaryWriter(log_dir = os.path.join(log_path,trainID))
os.makedirs(save_path, exist_ok=True)
logname = os.path.join(save_path, 'log_'+trainID+'.txt')

logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
torch.manual_seed(0)

traindata = glob('/pool/alex/Motion-Correction-3D/samples/output/train-new/*.nii.gz')
valdata = glob('/pool/alex/Motion-Correction-3D/samples/output/val-new/*.nii.gz')

batch_size_ = 10 ## 10
patches = 10
channels = 1
# 
size_ = 256
sigma_range = (0.01,5.0)
orientation = 3 ## 0 original
modalityMotion = 2  # 0 only reality motion, 1 only TorchIO, 2 combined reality+TIO
level_noise = 0.025
num_ghosts=5,
axes=2,
intensity=0.75,
restore=0.02,
degrees=10,
translation=10,
num_transforms=10,

trainset = MoCoDatasetRegressionUpdated(
                       traindata, 
                       patches=patches, 
                       size=size_,
                       orientation=orientation,
                       modalityMotion=modalityMotion, 
                       sigma_range=sigma_range,
                       level_noise=level_noise,
                       num_ghosts=num_ghosts,
                       axes=axes,
                       intensity=intensity,
                       restore=restore,
                       degrees=degrees,
                       translation=translation,
                       num_transforms=num_transforms,
                       image_interpolation='linear',
                       transform=None)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, sampler=None, shuffle=True)

valset = MoCoDatasetRegressionUpdated(
                       valdata, 
                       patches=patches, 
                       size=size_,
                       orientation=orientation,
                       modalityMotion=modalityMotion, 
                       sigma_range=sigma_range,
                       level_noise=level_noise,
                       num_ghosts=num_ghosts,
                       axes=axes,
                       intensity=intensity,
                       restore=restore,
                       degrees=degrees,
                       translation=translation,
                       num_transforms=num_transforms,
                       image_interpolation='linear',
                       transform=None)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_, sampler=None, shuffle=True)

### ----------------------------------------------------- ###
import torchvision.models as models
model = models.resnet101(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_classes = 1
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
model.to(device)
### ----------------------------------------------------- ###
start_epoch = 0
num_epochs= 2000
learning_rate = 1e-3 # 3e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scaler = GradScaler(enabled=True)
log_freq = 2
save_freq = 1
checkpoint = ""

ploss_level = math.inf
ploss_type = "L1"
loss_func = nn.MSELoss()

if checkpoint:
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'])
    optimizer.load_state_dict(chk['optimizer'])
    scaler.load_state_dict(chk['AMPScaler'])  
    best_loss = chk['best_loss']  
    start_epoch = chk['epoch'] + 1
else:
    start_epoch = 0
    best_loss = float('inf')

### ----------------------------------------------------- ###
for epoch in range(start_epoch, num_epochs):
    ### --- Train --- ###
    model.train()
    runningLoss = []
    train_loss = []
    print('Epoch '+ str(epoch)+ ': Train')
    for idx, (cor, img, ssimtmp) in enumerate(tqdm(train_loader)):
        cor = torch.unsqueeze(cor.to(device),1).float()
        img = torch.unsqueeze(img.to(device),1)
        ssimtmp = torch.unsqueeze(ssimtmp.to(device),1).float()
        cor = torch.reshape(cor, (batch_size_*patches, channels, size_,size_))
        img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
        ssimtmp = torch.reshape(ssimtmp, (batch_size_*patches, channels))
        
        optimizer.zero_grad()

        pred = model(cor)
        loss = loss_func(pred, ssimtmp)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(loss)
        runningLoss.append(loss)
        logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, idx, len(train_loader), loss))
        
        ##For tensorboard
        if idx % log_freq == 0:
            niter = epoch*len(train_loader)+idx
            tb_writer.add_scalar('Train/Loss', median(runningLoss), niter)
            tensorboard_regression(tb_writer,  cor[0, ...], img[0, ...], epoch, 'train')
            runningLoss = []
    
    if epoch % save_freq == 0:            
        checkpoint = {
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'AMPScaler': scaler.state_dict()         
                }
        torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
                
    tb_writer.add_scalar('Train/EpochLoss', median(train_loss), epoch)
    ### ------  validation ------  ####
    if val_loader:
        model.eval()
        with torch.no_grad():
            runningLoss = []
            val_loss = []
            runningAcc = []
            val_acc = []
            print('Epoch '+ str(epoch)+ ': Val')
            for i, (cor, img, ssimtmp) in enumerate(tqdm(val_loader)):
                cor = torch.unsqueeze(cor.to(device),1).float()
                img = torch.unsqueeze(img.to(device),1)
                ssimtmp = torch.unsqueeze(ssimtmp.to(device),1).float()
                cor = torch.reshape(cor, (batch_size_*patches, channels, size_,size_))
                img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
                ssimtmp = torch.reshape(ssimtmp, (batch_size_*patches, channels))
        
                pred = model(cor)
                loss = loss_func(pred, ssimtmp)
                
                val_loss.append(loss)
                runningLoss.append(loss)
                
                logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))

                #For tensorboard
                if i % log_freq == 0:
                    niter = epoch*len(val_loader)+i
                    tb_writer.add_scalar('Val/Loss', median(runningLoss), niter)
                    
                    tensorboard_regression(tb_writer,  cor[0, ...], img[0, ...], epoch, 'validation')
                    runningLoss = []
                    runningAcc = []
            if median(val_loss) < best_loss:
                best_loss = median(val_loss)
                checkpoint = {
                            'epoch': epoch,
                            'best_loss': best_loss,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'AMPScaler': scaler.state_dict()         
                        }
                torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
                
        tb_writer.add_scalar('Val/EpochLoss', median(val_loss), epoch)

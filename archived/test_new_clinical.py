import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './utils')
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

from utilities_new import MoCoDatasetRegressionClinicalNoGT


device = torch.device("cuda:0") 
log_path = './TBLogs'
trainID = 'TEST-RN18-Regression-TIO-RealityMot-Combined'
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

testdata = glob('/pool/alex/Motion-Correction-3D/samples/output/testDataClinical/*.nii.gz')
checkpoint = './Results/TEST-RN18-Regression-TIO-RealityMot-Combined_best.pth.tar'
batch_size_ = 10 
patches = 10
channels = 1
# 
size_ = 256
sigma_range = (0.01,2.5)
orientation = 0 # original
level_noise = 0.025
testset = MoCoDatasetRegressionClinicalNoGT(testdata, patches=patches, 
                       size=size_,
                       orientation=orientation,
                       level_noise= level_noise,
                       transform=None)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_, sampler=None, shuffle=False)

### ----------------------------------------------------- ###
import torchvision.models as models
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_classes = 1
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
model.to(device)

chk = torch.load(checkpoint, map_location=device)
model.load_state_dict(chk['state_dict'] )
trained_epoch = chk['epoch']
model.eval()
### ----------------------------------------------------- ###
start_epoch = 0
num_epochs= 2000
learning_rate = 3e-4
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
    model.load_state_dict(chk['state_dict'] )
    optimizer.load_state_dict(chk['optimizer'] )
    scaler.load_state_dict(chk['AMPScaler'] )  
    best_loss = chk['best_loss']   
    start_epoch = chk['epoch']  + 1
else:
    start_epoch = 0
    best_loss = float('inf')

### ----------------------------------------------------- ###
with torch.no_grad():
    for idx, (img) in enumerate(tqdm(test_loader)): 
        img = torch.unsqueeze(img.to(device),1)
        img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
        pred = model(img.float())
        print(pred.detach().cpu().numpy())
        niifiles = np.zeros((size_,size_, batch_size_*patches))
        for ii in range(0, batch_size_*patches):
            gt_ = img[ii,...].detach().cpu().numpy().squeeze()
            
            niifiles[:,:,ii] = gt_
        filled_img = nib.Nifti1Image(niifiles, np.eye(4))
        nib.save(filled_img, 'Tentative-Clinical'+str(idx)+'.nii.gz')

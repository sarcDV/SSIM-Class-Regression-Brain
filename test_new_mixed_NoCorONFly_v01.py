import os, sys
import random

# --------
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# --------
import nibabel as nib
from skimage.transform import resize
from skimage.metrics import structural_similarity

def main():
    """run with the following command:
       python test-clinical-evaluate-nii.py nii-file (file.nii.gz)
    """
    test_new_mixed_NoCorONFly(sys.argv[1],sys.argv[2],sys.argv[3])
    return

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg 


###########################################################################

def test_new_mixed_NoCorONFly(filein, chkin, neuralmodel):
    """ 
    python test_new_mixed_NoCorONFly_v01.py testdata-RN18-WConAug.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18
    """
    ### ----------------------------------------------------- ###
    device = torch.device("cuda:1") 
    checkpoint = chkin
    # checkpoint = './Results/TEST-RN18-Regression-TIO-RealityMot-Combined_best.pth.tar'
    # checkpoint = './Results/TEST-RN18-Regression-TIO-RealityMot-Combined-NoContrastAug_best.pth.tar'
    batch_size_, patches, channels, size_= 1,1,1,256
    level_noise = 0.025
    ### ----------------------------------------------------- ###
    if neuralmodel == 'RN18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    else:
        model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.to(device)
    
    chk = torch.load(checkpoint, map_location=device)
    model.load_state_dict(chk['state_dict'] )
    model.eval()
    ### ----------------------------------------------------- ###
    print("Evaluating: " + str(filein))
    a = nib.load(filein).get_fdata()
    with torch.no_grad():
        for ii in range(0, a.shape[2]):
            inimg = a[:,2*size_:,ii]
            gtimg = a[:,size_:2*size_,ii]
            ssimtmp = structural_similarity(gtimg, inimg, data_range=1)
            img = torch.unsqueeze(torch.tensor(inimg).to(device),1)
            img = torch.reshape(img, (batch_size_*patches, channels, size_,size_))
            pred = model(img.float())
            # print("Predicted SSIM value for slice: "+str(ii+1)+\
            #       " "+str(pred[0][0].detach().cpu().numpy())+" "+str(ssimtmp))
            # print(str(ii+1)+" "+str(ssimtmp)+\
            #       " "+str(pred[0][0].detach().cpu().numpy()))
            print(str(ssimtmp)+" "+str(pred[0][0].detach().cpu().numpy()))


if __name__ == "__main__":
	main()


# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-WConAug.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18 > RN18-WCA-batch-1.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-NConAug.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18 > RN18-WCA-batch-2.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-WConAug.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18 > RN18-WCA-batch-3.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-NConAug.nii.gz Checkpoints/RN18-WConAug.pth.tar RN18 > RN18-WCA-batch-4.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-WConAug.nii.gz Checkpoints/RN18-NConAug.pth.tar RN18 > RN18-NCA-batch-1.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-NConAug.nii.gz Checkpoints/RN18-NConAug.pth.tar RN18 > RN18-NCA-batch-2.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-WConAug.nii.gz Checkpoints/RN18-NConAug.pth.tar RN18 > RN18-NCA-batch-3.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-NConAug.nii.gz Checkpoints/RN18-NConAug.pth.tar RN18 > RN18-NCA-batch-4.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-WConAug.nii.gz Checkpoints/RN101-WConAug.pth.tar RN101 > RN101-WCA-batch-1.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-NConAug.nii.gz Checkpoints/RN101-WConAug.pth.tar RN101 > RN101-WCA-batch-2.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-WConAug.nii.gz Checkpoints/RN101-WConAug.pth.tar RN101 > RN101-WCA-batch-3.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-NConAug.nii.gz Checkpoints/RN101-WConAug.pth.tar RN101 > RN101-WCA-batch-4.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-WConAug.nii.gz Checkpoints/RN101-NConAug.pth.tar RN101 > RN101-NCA-batch-1.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN18-NConAug.nii.gz Checkpoints/RN101-NConAug.pth.tar RN101 > RN101-NCA-batch-2.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-WConAug.nii.gz Checkpoints/RN101-NConAug.pth.tar RN101 > RN101-NCA-batch-3.csv
# python test_new_mixed_NoCorONFly_v01.py testdata-RN101-NConAug.nii.gz Checkpoints/RN101-NConAug.pth.tar RN101 > RN101-NCA-batch-4.csv

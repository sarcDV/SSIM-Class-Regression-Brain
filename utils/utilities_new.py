import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import nibabel as nib
import random
from numpy.lib.arraypad import pad
import torch
from statistics import median
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from skimage import exposure
from progressbar import ProgressBar
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)
import torchvision.utils as vutils
import torchio
import time
import multiprocessing.dummy as multiprocessing
from tqdm import tqdm
###########################################################################
##### Auxiliaries  ########################################################
###########################################################################

def pad3D(invol, max1d, max2d, max3d):
    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2),
             ((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2),
             ((max3d-invol.shape[2])//2, (max3d-invol.shape[2])//2)), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')
    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')
    if aa.shape[2] == (int(max3d)-1):
        aa = np.pad(aa, ((0,0),(0,0),(1,0)), 'constant')

    return aa

def pad2D(invol, max1d, max2d):
    if (invol.shape[0] % 2) != 0:  # if shape[0] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((1,invol.shape[1]))), axis=0)
        
    if (invol.shape[1] % 2) != 0:  # if shape[1] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((invol.shape[0],1))), axis=1)

    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2),
             ((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2)), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')
    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')

    return aa

def pad2DD1(invol, max1d):
    if (invol.shape[0] % 2) != 0:  # if shape[0] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((1,invol.shape[1]))), axis=0)

    aa = np.pad(invol, 
            (((max1d-invol.shape[0])//2, (max1d-invol.shape[0])//2 ) ), 
            'constant')

    if aa.shape[0] == (int(max1d)-1):
        aa = np.pad(aa, ((1,0),(0,0),(0,0)), 'constant')

    return aa

def pad2DD2(invol, max2d):
    if (invol.shape[1] % 2) != 0:  # if shape[1] is dispari  (odd)
        invol = np.concatenate((invol, np.zeros((invol.shape[0],1))), axis=1)

    aa = np.pad(invol, 
            (((max2d-invol.shape[1])//2, (max2d-invol.shape[1])//2)), 
            'constant')

    if aa.shape[1] == (int(max2d)-1):
        aa = np.pad(aa, ((0,0),(1,0),(0,0)), 'constant')

    return aa

def randomCrop(img, cor, width, height): 
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    cor = cor[y:y+height, x:x+width]
    return img, cor

def randomCropInput(img, width, height): 
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    
    return img

def PadAndResize(img, width, height):
    dsize = (width, height)
    if img.shape[0]==img.shape[1]:
        img  = resize(img, dsize)
    elif img.shape[0]>img.shape[1]:
        img = pad2DD2(img, img.shape[0])
        img = resize(img, dsize)
    elif img.shape[0]<img.shape[1]:
        img = pad2DD1(img, img.shape[1])
        img = resize(img, dsize)
    return img

def ResizeAndPadVolume(img, width, height):
    """2d resize and padding """
    dsize = (width, height)
    imgdsize_ = np.zeros((width, height, img.shape[2]))
    if img.shape[0] == img.shape[1]:
        for ii in range(0, img.shape[2]):
            imgdsize_[:,:,ii] = resize(img[:,:,ii], dsize)
    else:
        ## find largest in-plane size:
        maxdim_ = np.argmax((img.shape[0],img.shape[1]))
        if maxdim_ == 0:
            ## calculate the ratio:
            dsize0_ = (width, int((width/img.shape[0])*img.shape[1]))
            for ii in range(0,img.shape[2]):
                imgdsize_[:,:,ii]= pad2D(resize(img[:,:,ii], dsize0_), width, height)
        elif maxdim_ == 1:
            ## calculate the ratio:
            dsize1_ = (int((height/img.shape[1])*img.shape[0]), height )
            for ii in range(0,img.shape[2]):
                imgdsize_[:,:,ii]=pad2D(resize(img[:,:,ii], dsize1_), width, height)

    ## imgdsize_ = pad3D(imgdsize_,  width, height, width)
    return imgdsize_

def checkSIZE(img, width, height):
    if img.shape[0] < width or img.shape[1] < height :
        img = pad2D(img, width, height)

    return img
###########################################################################
##### Contrast Augmentation  ##############################################
###########################################################################

def randomContrastAug(img):
    expo_selection = np.random.randint(0,5,1)
    if expo_selection[0] == 0:
        adjimg = exposure.adjust_gamma(img, np.random.uniform(0.75, 1.75, 1)[0])
    elif expo_selection[0] == 1:
        adjimg = exposure.equalize_adapthist(img, 
											kernel_size=int(np.random.randint(25, high=100, size=(1))[0]), #21, 
											clip_limit=0.01, 
											nbins=512)
    elif expo_selection[0] == 2:
        adjimg = exposure.adjust_sigmoid(img, 
	 								   cutoff=np.random.uniform(0.01, 0.75, 1)[0], #0.5, 
	  								   gain=int(np.random.randint(1, high=4, size=(1))[0]), #10, 
	  								   inv=False)
    elif expo_selection[0] == 3:
       adjimg = np.abs(exposure.adjust_log(img, np.random.uniform(-0.5, 0.5, 1)[0]))
    else:
        adjimg = img

    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg #, expo_selection[0]

###########################################################################
##### Cut noise level  ####################################################
###########################################################################

def cutNoise(img, level):
    adjimg = (img>level)*1.0*img
    ## normalize again:
    adjimg = (adjimg-adjimg.min())/(adjimg.max()+1e-16-adjimg.min())

    return adjimg 

###########################################################################
##### Motion  #############################################################
###########################################################################

class Motion2DOld():
    def __init__(self, sigma_range=(0.10, 2.5), n_threads=10):
        self.sigma_range = sigma_range
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        rot = self.sigma*random.randint(-1,1)
        img_aux = ndimage.rotate(self.img, rot, reshape=False)
        # rot = np.random.uniform(self.mu, self.sigma, 1)*random.randint(-1,1)
        # rot = np.random.normal(self.mu, self.sigma, 1)*random.randint(-1,1)
        # img_aux = ndimage.rotate(self.img, rot[0], reshape=False)
        img_h = np.fft.fft2(img_aux)
        if self.axis_selection == 0:
            self.aux[:,idx]=img_h[:,idx]
        else:
            self.aux[idx,:]=img_h[idx,:]

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]
        self.mu=0
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]))
        else:
            for idx in range(self.aux.shape[1] if self.axis_selection == 0 else self.aux.shape[0]):
                self.__perform_singlePE(idx)
        cor =np.abs(np.fft.ifft2(self.aux)) 
        del self.img, self.aux, self.axis_selection, self.mu, self.sigma
        return cor/(cor.max()+1e-16)

class Motion2D():
    def __init__(self, sigma_range=(0.10, 2.5), restore_original=5e-2, n_threads=10):
        self.sigma_range = sigma_range
        self.restore_original = restore_original
        self.n_threads = n_threads

    def __perform_singlePE(self, idx):
        img_aux = ndimage.rotate(self.img, self.random_rots[idx], reshape=False)
        img_h = np.fft.fft2(img_aux)            
        if self.axis_selection == 0:
            self.aux[:,self.portion[idx]]=img_h[:,self.portion[idx]]  
        else:
            self.aux[self.portion[idx],:]=img_h[self.portion[idx],:]  

    def __call__(self, img):
        self.img = img
        self.aux = np.zeros(img.shape) + 0j
        self.axis_selection = np.random.randint(0,2,1)[0]

        if self.axis_selection == 0:
            dim = 1
        else:
            dim = 0

        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[dim]//n_), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[dim]//n_), 
                                                        high=int(img.shape[dim]), 
                                                        size=int(img.shape[dim]//2*n_), dtype=int))) 
            self.portion = np.concatenate((portiona, portionb))  
        else:
            self.portion = np.sort(np.unique(np.random.randint(low=int(img.shape[dim]//2)-int(img.shape[dim]//n_+1), 
                                                     high=int(img.shape[dim]//2)+int(img.shape[dim]//n_+1), 
                                                     size=int(img.shape[dim]//n_+1), dtype=int)))
        self.sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1], 1)[0]
        self.random_rots = self.sigma * np.random.randint(-1,1,len(self.portion))
        #  self.random_rots = np.random.randint(-4,4,len(self.portion))

        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(len(self.portion)-1))
        else:
            for idx in range(len(self.portion)-1):
                self.__perform_singlePE(idx)     
        cor =np.abs(np.fft.ifft2(self.aux)) # + self.restore_original *img

        del self.img, self.aux, self.axis_selection, self.portion, self.random_rots
        return cor/(cor.max()+1e-16)
###########################################################################
##### slice selection  ####################################################
###########################################################################

def select_slice_orientation(test, orientation):
    if orientation == 3:
        if test.shape[2] > (test.shape[0]//2):
            rnd_orient = np.random.randint(0,3,1)[0]
            # print(rnd_orient)
            if rnd_orient == 0:
                rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)
                
                img = (test[:,rndslice_[0],:])
            elif rnd_orient == 1:
                rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
                                        
                img = np.rot90(test[:,:,rndslice_[0]])    
            else:
                rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
                img = np.flipud(test[rndslice_[0],:,:])
        else:
            rnd_orient = 1
            rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)                           
            img = np.rot90(test[:,:,rndslice_[0]]) 
    
    elif orientation == 0:
        rnd_orient = 0 
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
    elif orientation == 1:
        rnd_orient = 1
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)       
        img = (test[:,rndslice_[0],:])
    elif orientation == 2: 
        rnd_orient = 2
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
        img = np.flipud(test[rndslice_[0],:,:])

    img = (img-img.min())/(img.max()+1e-16-img.min())
            
    return img, rndslice_, rnd_orient

def select_slice_orientation_both(test, cor,orientation):

    if orientation == 3:
        if test.shape[2] > (test.shape[0]//2):
            rnd_orient = np.random.randint(0,3,1)[0]
            # print(rnd_orient)
            if rnd_orient == 0:
                rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)
                
                img = (test[:,rndslice_[0],:])
                imgcor = (cor[:,rndslice_[0],:])
            elif rnd_orient == 1:
                rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
                                        
                img = np.rot90(test[:,:,rndslice_[0]]) 
                imgcor = np.rot90(cor[:,:,rndslice_[0]])   
            else:
                rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
                img = np.flipud(test[rndslice_[0],:,:])
                imgcor = np.flipud(cor[rndslice_[0],:,:])
        else:
            rnd_orient = 1
            rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)                           
            img = np.rot90(test[:,:,rndslice_[0]]) 
            imgcor = np.rot90(cor[:,:,rndslice_[0]]) 
    
    elif orientation == 0:
        rnd_orient = 0 
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                            high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                            size=1)
        img = np.rot90(test[:,:,rndslice_[0]]) 
        imgcor = np.rot90(cor[:,:,rndslice_[0]]) 
    elif orientation == 1:
        rnd_orient = 1
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                            high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                            size=1)       
        img = (test[:,rndslice_[0],:])
        imgcor = (cor[:,rndslice_[0],:])
    elif orientation == 2: 
        rnd_orient = 2
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                            high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                            size=1)                            
        img = np.flipud(test[rndslice_[0],:,:])
        imgcor = np.flipud(cor[rndslice_[0],:,:])

    img = (img-img.min())/(img.max()+1e-16-img.min())
    imgcor = (imgcor-imgcor.min())/(imgcor.max()+1e-16-imgcor.min())       
    return img, imgcor, rndslice_, rnd_orient
###########################################################################
##### Motion Corruption Class  ############################################
###########################################################################
class MoCoDataset2DNoPatchesBsOne():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                orientation=3,
                sigma_range=(0.01, 2.5),
                level_noise=0.07,
                transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.orientation = orientation
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.transform = transform
        self.cter = Motion2DOld(n_threads=10)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        img, slice, orient = select_slice_orientation(image_in_, self.orientation)
        img = randomContrastAug(cutNoise(img, self.level_noise))
        cor = self.cter(img)
        ssimtmp = structural_similarity(img, cor, data_range=1)
        ### ------        
        if self.transform:
            img = self.transform(img)
            cor = self.transform(cor)
            ssimtmp = self.transform(ssimtmp)
        
        return cor, img, ssimtmp

class MoCoDatasetRegressionClinicalNoGT():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=0,
                       level_noise=0.02,
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.transform = transform
        self.level_noise = level_noise

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        size_ = self.size
        stackimg = np.zeros((self.patches, size_, size_))

        for ii in range(0, self.patches):
            # img, slice, orient = select_slice_orientation(image_in_, self.orientation)
            slice = np.random.randint(low=0,high=image_in_.shape[2], size=1)[0]
            img = image_in_[:,:,slice]
            ## img = checkSIZE(img, size_, size_)
            img = cutNoise(img, self.level_noise)
            img = PadAndResize(img, size_, size_)
            img = checkSIZE(img, size_, size_)
            # if (img.shape[0]<=size_) and (img.shape[1]<=size_):
            #     img = pad2D(img, size_, size_)       
            # elif (img.shape[0]>size_) and (img.shape[1]>size_):
            #     img = PadAndResize(img, width=size_, height=size_)
            # elif (img.shape[0]<=size_) and (img.shape[1]>size_):
            #     img = PadAndResize(img, width=size_, height=size_)
            # elif (img.shape[0]>size_) and (img.shape[1]<=size_):
            #     img = PadAndResize(img, width=size_, height=size_)
            # else:
            #     print(img.shape, 'ciao')
            ### ------- 
            
            stackimg[ii,:,:]= img
        
        if self.transform:           
            stackimg = self.transform(stackimg)
            
        return stackimg

class MoCoDatasetRegressionUpdated():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=3,
                       modalityMotion=2, # 0 only reality motion, 1 only TorchIO, 2 combined reality+TIO
                       sigma_range=(0.0, 3.0),
                       contrastAug = True,
                       level_noise=0.025,
                       num_ghosts=5,
                       axes=2,
                       intensity=0.75,
                       restore=0.02,
                       degrees=10,
                       translation=10,
                       num_transforms=10,
                       image_interpolation='linear',
                       transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.modalityMotion = modalityMotion
        self.transform = transform
        self.sigma_range = sigma_range
        self.contrastAug = contrastAug
        self.level_noise = level_noise
        self.num_ghosts = num_ghosts
        self.axes = axes
        self.intensity = intensity
        self.restore = restore
        self.degrees = degrees
        self.translation = translation
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation
        self.cter = Motion2DOld(n_threads=10, sigma_range=self.sigma_range)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackorig = np.zeros((self.patches, size_, size_))
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))
        trans = torchio.transforms.RandomGhosting(num_ghosts=int(np.random.randint(low=3,high=self.num_ghosts, size=1)[0]),#5,
                                         axes=np.random.randint(self.axes, size=1)[0],
                                         intensity=np.random.uniform(0.05, self.intensity, 1)[0],# 1.75,
                                         restore=np.random.uniform(0.01, self.restore, 1)[0])# 0.02)
        transMot = torchio.transforms.RandomMotion(degrees=np.random.uniform(0.01, self.degrees, 1)[0],# 10,
                                         translation=np.random.uniform(0.01, self.translation, 1)[0],# 10,
                                         num_transforms=int(np.random.randint(low=2,high=self.num_transforms, size=1)[0]),#5,
                                         image_interpolation='linear')
        image_in_ = ResizeAndPadVolume(image_in_, width=size_, height=size_)

        # for ii in range(0, self.patches):
        if self.modalityMotion == 2:
            kk = np.random.randint(low=1, high=100, size=1)[0]
            for ii in range(0, self.patches):
                if (kk % 2) != 0:  
                    img, slice, orient = select_slice_orientation(image_in_, self.orientation)
                    img = checkSIZE(img, size_, size_)
                    original = img
                    ## contrast augmentation & noise reduction/background only:
                    if self.contrastAug == True:
                        img = randomContrastAug(cutNoise(img, self.level_noise)) 
                    else:
                        img = cutNoise(img, self.level_noise) # no contrast enhancement
                    
                    cor = self.cter(img)
                    ssimtmp = structural_similarity(img, cor, data_range=1)
                    stackorig[ii,:,:] = original
                    stackcor[ii,:,:]= cor
                    stackimg[ii,:,:]= img
                    stackssim[ii,0] = ssimtmp
                else:
                    img, slice, orient = select_slice_orientation(image_in_, self.orientation)
                    img = checkSIZE(img, size_, size_)
                    original = img
                    ## contrast augmentation & noise reduction/background only:
                    if self.contrastAug == True:
                        img = randomContrastAug(cutNoise(img, self.level_noise)) 
                    else:
                        img = cutNoise(img, self.level_noise) # no contrast enhancement
                    
                    ## corrupt with torchio:
                    trans = torchio.transforms.RandomGhosting(num_ghosts=int(np.random.randint(low=3,high=self.num_ghosts, size=1)[0]),#5,
                                         axes=np.random.randint(self.axes, size=1)[0],
                                         intensity=np.random.uniform(0.05, self.intensity, 1)[0],# 1.75,
                                         restore=np.random.uniform(0.01, self.restore, 1)[0])# 0.02)
                    transMot = torchio.transforms.RandomMotion(degrees=np.random.uniform(0.01, self.degrees, 1)[0],# 10,
                                         translation=np.random.uniform(0.01, self.translation, 1)[0],# 10,
                                         num_transforms=int(np.random.randint(low=2,high=self.num_transforms, size=1)[0]),#5,
                                         image_interpolation='linear')
                    testtens = torch.unsqueeze(torch.unsqueeze(torch.tensor(img),0),0)
                    d_ = transMot(testtens)
                    d_ = trans(d_)
                    cor = d_.detach().cpu().numpy().squeeze()
                    ssimtmp = structural_similarity(img, cor, data_range=1)
                    stackorig[ii,:,:] = original
                    stackcor[ii,:,:]= cor
                    stackimg[ii,:,:]= img
                    stackssim[ii,0] = ssimtmp
                    
        elif self.modalityMotion == 0:
            for ii in range(0, self.patches):
                img, slice, orient = select_slice_orientation(image_in_, self.orientation)
                img = checkSIZE(img, size_, size_)
                original = img
                ## contrast augmentation & noise reduction/background only:
                if self.contrastAug == True:
                    img = randomContrastAug(cutNoise(img, self.level_noise)) 
                else:
                    img = cutNoise(img, self.level_noise) # no contrast enhancement
                
                cor = self.cter(img)
                ssimtmp = structural_similarity(img, cor, data_range=1)
                stackorig[ii,:,:] = original
                stackcor[ii,:,:]= cor
                stackimg[ii,:,:]= img
                stackssim[ii,0] = ssimtmp
        else:
            for ii in range(0, self.patches):
                img, slice, orient = select_slice_orientation(image_in_, self.orientation)
                img = checkSIZE(img, size_, size_)
                original = img
                ## contrast augmentation & noise reduction/background only:
                if self.contrastAug == True:
                    img = randomContrastAug(cutNoise(img, self.level_noise)) 
                else:
                    img = cutNoise(img, self.level_noise) # no contrast enhancement
                
                ## corrupt with torchio:
                trans = torchio.transforms.RandomGhosting(num_ghosts=int(np.random.randint(low=3,high=self.num_ghosts, size=1)[0]),#5,
                                         axes=np.random.randint(self.axes, size=1)[0],
                                         intensity=np.random.uniform(0.05, self.intensity, 1)[0],# 1.75,
                                         restore=np.random.uniform(0.01, self.restore, 1)[0])# 0.02)
                transMot = torchio.transforms.RandomMotion(degrees=np.random.uniform(0.01, self.degrees, 1)[0],# 10,
                                         translation=np.random.uniform(0.01, self.translation, 1)[0],# 10,
                                         num_transforms=int(np.random.randint(low=2,high=self.num_transforms, size=1)[0]),#5,
                                         image_interpolation='linear')
                testtens = torch.unsqueeze(torch.unsqueeze(torch.tensor(img),0),0)
                d_ = transMot(testtens)
                d_ = trans(d_)
                cor = d_.detach().cpu().numpy().squeeze()
                ssimtmp = structural_similarity(img, cor, data_range=1)
                stackorig[ii,:,:] = original
                stackcor[ii,:,:]= cor
                stackimg[ii,:,:]= img
                stackssim[ii,0] = ssimtmp

        if self.transform:
            stackorig = self.transform(stackorig)
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            
        return stackcor, stackimg, stackssim, stackorig

###########################################################################
###########################################################################
###########################################################################
def tensorboard_regression(writer, inputs, outputs, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def tensorboard_correction(writer, inputs, outputs, targets, epoch, section='train'):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, ...],
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                        vutils.make_grid(inputs[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                        vutils.make_grid(targets[0, ...],
                                        normalize=True,
                                        scale_each=True),
                        epoch)

def getSSIM(gt, out, gt_flag, data_range=1):
    vals = []
    for i in range(gt.shape[0]):
        if not gt_flag[i]:
            continue
        for j in range(gt.shape[1]):
            vals.append(structural_similarity(gt[i,j,...], out[i,j,...], data_range=data_range))
    return median(vals)

import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import random
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
    adjimg = (adjimg-adjimg.min())/(adjimg.max()-adjimg.min())

    return adjimg #, expo_selection[0]

###########################################################################
##### Motion  #############################################################
###########################################################################
def generate_motion_2d_old(img):
    aux = np.zeros([img.shape[0],img.shape[1]]) + 1j*np.zeros([img.shape[0],img.shape[1]])
    axis_selection = np.random.randint(0,2,1)
    mu=0
    sigma=np.random.uniform(0.10, 10.0, 1)[0]
    if axis_selection[0] == 0:
        for kk in range(0, aux.shape[1]):
            rot = np.random.normal(mu, sigma, 1)*random.randint(-1,1)
            img_aux = ndimage.rotate(img, rot[0], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[:,kk]=img_h[:,kk]        
        cor =np.abs(np.fft.ifft2(aux)) 
    else:
        for kk in range(0, aux.shape[0]):
            rot = np.random.normal(mu, sigma, 1)*random.randint(-1,1)
            img_aux = ndimage.rotate(img, rot[0], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[kk,:]=img_h[kk,:]        
        cor =np.abs(np.fft.ifft2(aux)) 
    return cor/(cor.max()+1e-16)

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
    def __init__(self, restore_original=5e-2, n_threads=20):
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
        self.random_rots = np.random.randint(-4,4,len(self.portion))

        if self.n_threads > 1:
            pool = multiprocessing.Pool(self.n_threads)
            pool.map(self.__perform_singlePE, range(len(self.portion)-1))
        else:
            for idx in range(len(self.portion)-1):
                self.__perform_singlePE(idx)     
        cor =np.abs(np.fft.ifft2(self.aux)) + self.restore_original *img

        del self.img, self.aux, self.axis_selection, self.portion, self.random_rots
        return cor/(cor.max()+1e-16)

def generate_motion_2d(img):
    aux = np.zeros([img.shape[0],img.shape[1]]) + 1j*np.zeros([img.shape[0],img.shape[1]])
    axis_selection = np.random.randint(0,2,1)
    
    if axis_selection[0] == 0:
        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[1]//n_), 
                                                        size=int(img.shape[1]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[1]//n_), 
                                                        high=int(img.shape[1]), 
                                                        size=int(img.shape[1]//2*n_), dtype=int))) 
            portion = np.concatenate((portiona, portionb))  
        else:
            portion = np.sort(np.unique(np.random.randint(low=int(img.shape[1]//2)-int(img.shape[1]//n_+1), 
                                                     high=int(img.shape[1]//2)+int(img.shape[1]//n_+1), 
                                                     size=int(img.shape[1]//n_+1), dtype=int)))
        random_rots = np.random.randint(-4,4,len(portion))
        
        for kk in range(0, len(portion)-1):
            img_aux = ndimage.rotate(img, random_rots[kk], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[:,portion[kk]]=img_h[:,portion[kk]]        
        cor =np.abs(np.fft.ifft2(aux)) + 0.05 *img
    else:
        n_ = np.random.randint(2,8,1)[0]
        intext_ = np.random.randint(0,2,1)[0]
        if intext_ == 0:
            portiona = np.sort(np.unique(np.random.randint(low=0, 
                                                        high=int(img.shape[0]//n_), 
                                                        size=int(img.shape[0]//2*n_), dtype=int)))
            portionb = np.sort(np.unique(np.random.randint(low=int((n_-1)*img.shape[0]//n_), 
                                                        high=int(img.shape[0]), 
                                                        size=int(img.shape[0]//2*n_), dtype=int))) 
            portion = np.concatenate((portiona, portionb))
        else:
            portion = np.sort(np.unique(np.random.randint(low=int(img.shape[0]//2)-int(img.shape[0]//n_), 
                                                     high=int(img.shape[0]//2)+int(img.shape[0]//n_), 
                                                     size=int(img.shape[0]//n_), dtype=int)))
        random_rots =  np.random.randint(-4,4,len(portion))
        for kk in range(0, len(portion)):
            img_aux = ndimage.rotate(img, random_rots[kk], reshape=False)
            img_h = np.fft.fft2(img_aux)
            aux[portion[kk],:]=img_h[portion[kk],:]        
        cor =np.abs(np.fft.ifft2(aux)) +  0.05 *img 
    return cor/(cor.max()+1e-16)


def corrupt_recursively(test):
    rnd_orient = np.random.randint(0,3,1)[0]
    # print(rnd_orient)
    if rnd_orient == 0:
        rndslice_ = np.random.randint(low=int(test.shape[1]//2)-int(test.shape[1]//4), 
                                    high=int(test.shape[1]//2)+int(test.shape[1]//4), 
                                    size=1)
        
        img = (test[:,rndslice_[0],:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    elif rnd_orient == 1:
        rndslice_ = np.random.randint(low=int(test.shape[2]//2)-int(test.shape[2]//4), 
                                    high=int(test.shape[2]//2)+int(test.shape[2]//4), 
                                    size=1)
                                   
        img = np.rot90(test[:,:,rndslice_[0]]) 
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)
    else:
        rndslice_ = np.random.randint(low=int(test.shape[0]//2)-int(test.shape[0]//4), 
                                    high=int(test.shape[0]//2)+int(test.shape[0]//4), 
                                    size=1)
                                    
        img = np.flipud(test[rndslice_[0],:,:])
        img = (img-img.min())/(img.max()-img.min())
        cor = generate_motion_2d(img)

    return img, cor, rndslice_, rnd_orient

def select_slice_orientation(test):
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

    img = (img-img.min())/(img.max()-img.min())
            
    return img, rndslice_, rnd_orient
###########################################################################
##### Motion Corruption Class  ############################################
###########################################################################
class MoCoDataset():
    """Motion Correction Dataset"""
    def __init__(self, input_list, patches=10, transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.patches = patches
        self.transform = transform
        self.cter = Motion2DOld(n_threads=20)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = 128
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))

        for ii in range(0, self.patches):
            img, slice, orient = select_slice_orientation(image_in_)
            if (img.shape[0]<=size_) and (img.shape[1]<=size_):
                img = pad2D(img, size_, size_)       
            elif (img.shape[0]>size_) and (img.shape[1]>size_):
                img = randomCropInput(img, width=size_, height=size_)
            elif (img.shape[0]<=size_) and (img.shape[1]>size_):
                img = pad2DD1(img, size_) 
                img = randomCropInput(img,  width=size_, height=size_)
            elif (img.shape[0]>size_) and (img.shape[1]<=size_):
                img = pad2DD2(img, size_) 
                img = randomCropInput(img,  width=size_, height=size_)
            else:
                print(img.shape, 'ciao')
            ### ------- 
            img = randomContrastAug(img)
            # cor = generate_motion_2d(img)
            # cor = generate_motion_2d_old(img)
            # cter = Motion2DOld(n_threads=20)
            cor = self.cter(img)
            ssimtmp = structural_similarity(img, cor, data_range=1)

            stackcor[ii,:,:]= cor
            stackimg[ii,:,:]= img
            stackssim[ii,0] = ssimtmp
        
        if self.transform:
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            # img = self.transform(img)
            # cor = self.transform(cor)
            # ssimtmp = self.transform(ssimtmp)
        
        return stackcor, stackimg, stackssim
        # return cor, img, ssimtmp

class MoCoDataset2DNoPatchesBsOne():
    """Motion Correction Dataset"""
    def __init__(self, input_list, transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.transform = transform
        self.cter = Motion2DOld(n_threads=20)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        img, slice, orient = select_slice_orientation(image_in_)
        ### ------- 
        img = randomContrastAug(img)
        # cor = generate_motion_2d(img)
        
        cor = self.cter(img)
        # cor = generate_motion_2d_old(img)
        ssimtmp = structural_similarity(img, cor, data_range=1)

        
        if self.transform:
            img = self.transform(img)
            cor = self.transform(cor)
            ssimtmp = self.transform(ssimtmp)
        
        return cor, img, ssimtmp


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

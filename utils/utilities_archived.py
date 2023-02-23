def checkSIZE(img, width, height):
    if img.shape[0] < width or img.shape[1] < height :
        img = pad2D(img, width, height)

    return img

class MoCoDataset():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=0,
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
        self.patches = patches
        self.size = size
        self.orientation = orientation
        self.transform = transform
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.cter = Motion2DOld(n_threads=10)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))

        for ii in range(0, self.patches):
            img, slice, orient = select_slice_orientation(image_in_, self.orientation)
            img = randomContrastAug(cutNoise(img, self.level_noise))
            cor = self.cter(img)
            if (img.shape[0]<=size_) and (img.shape[1]<=size_):
                img = pad2D(img, size_, size_)
                cor = pad2D(cor, size_, size_)       
            elif (img.shape[0]>size_) and (img.shape[1]>size_):
                img, cor = randomCrop(img, cor, width=size_, height=size_) 
                # img = randomCropInput(img, width=size_, height=size_)
            elif (img.shape[0]<=size_) and (img.shape[1]>size_):
                img = pad2DD1(img, size_) 
                cor = pad2DD1(cor, size_)
                img, cor = randomCrop(img, cor, width=size_, height=size_) 
                # img = pad2DD1(img, size_) 
                # img = randomCropInput(img,  width=size_, height=size_)
            elif (img.shape[0]>size_) and (img.shape[1]<=size_):
                img = pad2DD2(img, size_) 
                cor = pad2DD2(cor, size_) 
                img, cor = randomCrop(img, cor, width=size_, height=size_)
                # img = pad2DD2(img, size_) 
                # img = randomCropInput(img,  width=size_, height=size_)
            else:
                print(img.shape, 'ciao')
            ### ------- 
            # img = randomContrastAug(img)
            # cor = generate_motion_2d(img)
            # cor = generate_motion_2d_old(img)
            # cter = Motion2DOld(n_threads=10)
            # cor = self.cter(img)
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
        

class MoCoDatasetRegression():
    """Motion Correction Dataset"""
    def __init__(self, input_list, 
                       patches=10, 
                       size=256,
                       orientation=0,
                       sigma_range=(0.01, 2.5),
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
        self.sigma_range = sigma_range
        self.level_noise = level_noise
        self.cter = Motion2DOld(n_threads=10)

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        
        size_ = self.size
        stackcor = np.zeros((self.patches, size_, size_))
        stackimg = np.zeros((self.patches, size_, size_))
        stackssim = np.zeros((self.patches, 1))

        for ii in range(0, self.patches):
            img, slice, orient = select_slice_orientation(image_in_, self.orientation)
            img = randomContrastAug(cutNoise(img, self.level_noise))
            
            if (img.shape[0]<=size_) and (img.shape[1]<=size_):
                img = pad2D(img, size_, size_)       
            elif (img.shape[0]>size_) and (img.shape[1]>size_):
                img = PadAndResize(img, width=size_, height=size_)
            elif (img.shape[0]<=size_) and (img.shape[1]>size_):
                img = PadAndResize(img, width=size_, height=size_)
            elif (img.shape[0]>size_) and (img.shape[1]<=size_):
                img = PadAndResize(img, width=size_, height=size_)
            else:
                print(img.shape, 'ciao')
            ### ------- 
            cor = self.cter(img)
            ssimtmp = structural_similarity(img, cor, data_range=1)

            stackcor[ii,:,:]= cor
            stackimg[ii,:,:]= img
            stackssim[ii,0] = ssimtmp
        
        if self.transform:
            stackcor = self.transform(stackcor)
            stackimg = self.transform(stackimg)
            stackssim = self.transform(stackssim)
            
        return stackcor, stackimg, stackssim

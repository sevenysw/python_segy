import os
import cv2
import glob
import h5py
import segyio
import time
from torch.utils.data import Dataset
import torch
import numpy as np
from progressbar import *
from gain import *
from download_data import *
import matplotlib.pyplot as plt
import random

class DownsamplingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean data patches
        rate: data sampling rate when regular=False, e.g. 0.3
              data sampling interval when regular=True
    """
    def __init__(self, xs, rate, regular = False):
        super(DownsamplingDataset, self).__init__()
        self.xs = xs
        self.rate = rate
        self.regular = regular

    def __getitem__(self, index):
        batch_x = self.xs[index]
        # the type of the data must be tensor
        if self.regular:
            mask = regular_mask(batch_x,self.rate)
        else:
            mask = irregular_mask(batch_x,self.rate)
        batch_y = mask.mul(batch_x)
                
        return batch_y, batch_x,mask

    def __len__(self):
        return self.xs.size(0)


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean data patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)

def irregular_mask(data,rate):
    """the mask matrix of random sampling
    Args:
        data: original data patches
        rate: sampling rate,range(0,1)
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(),dtype=torch.float64) #tensor
    v = round(n*rate)
    TM = random.sample(range(n),v)
    mask[:,:,TM]=1 # missing by column 
    return mask

def regular_mask(data,a):

    """the mask matrix of regular sampling
    Args:
        data: original data patches
        a(int): sampling interval, e.g: a = 5, sampling like : 100001000010000
    """
    n = data.size()[-1]
    mask = torch.torch.zeros(data.size(),dtype=torch.float64)
    for i in range(n):
        if (i+1)%a==1:
            mask[:,:,i]=1
        else:
            mask[:,:,i]=0
    return mask 

def patch_show(train_data, save = False,root = ''):
    '''
    show some sampels of train data
    save: save or not save the showed sample
    root(path)：if save=True, the data will be saved to this path(as a .png picture)
    '''
    samples = 4
    idxs = np.random.choice(len(train_data), samples, replace=True)
    print(idxs)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, idx in enumerate(idxs):
        plt_idx = i+1
        data = train_data[idx]
        y,x = np.reshape(data[0],(data[0].shape[1],data[0].shape[2])),np.reshape(data[1],(data[1].shape[1],data[1].shape[2]))
        plt.subplot(2, samples, plt_idx)
        plt.imshow(x)
        plt.axis('off')
        plt.subplot(2, samples, plt_idx+samples)
        plt.imshow(y)        
        plt.axis('off')
    plt.show()

    if save:
        path = os.path.join(root,"samples.png")
        plt.savefig(path)

def data_aug(img, mode=None):
    # data augmentation
    if mode == 0:
        # original
        return img
    if mode == 1:
        # flip up and down
        return np.flipud(img)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(img)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(img))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(img, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(img, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(img, k=3))


def progress_bar(temp_size, total_size,patch_num,file,file_list):
    done = int(50 * temp_size / total_size)
#    sys.stdout.write("\r[%s%s][%s%s] %d%% %s" % (i+1,len(file_list),'#' * done, ' ' * (50 - done), 100 * temp_size / total_size,patch_num))
    sys.stdout.write("\r[%s/%s][%s%s] %d%% %s" % (file+1,file_list,'#' * done, ' ' * (50 - done), 100 * temp_size / total_size,patch_num))
    sys.stdout.flush()
'''

def progress_bar(temp_size, total_size):  
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10*total_size).start()
    pbar.update(10*temp_size+1) 
'''


def _compute_n_patches(i_h, i_w, p_h, p_w,s_h,s_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image width
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    s_h : int
        the moving step in the image height
    s_w: int
        the moving step in the image width
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    extraction_step：moving step
    """
    n_h = np.floor((i_h - p_h)/s_h)+1
    n_w = np.floor((i_w - p_w)/s_w)+1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _compute_total_patches(h, w, p_h, p_w,s_h,s_w,aug_times=[],scales=[],max_patches=None):
    num = 0
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        num += _compute_n_patches(h_scaled, w_scaled, p_h, p_w,s_h,s_w,max_patches=None)*(aug_times+1)
    return num


def gen_patches(data,patch_size =(64,64),stride = (32,32),file = 1,file_list = 1,total_patches_num=None,train_data_num=float('inf'),patch_num = None,aug_times=[],scales = [],q = None,single_patches_num=None,verbose=None):
    '''
    Args:
        aug_time(list): Corresponding function data_aug, if aug_time=[],mean don`t use the aug
        scales(list): data scaling; default scales = [],mean that the data don`t perform scaling,
                      if perform scaling, you can set scales=[0.9,0.8,...]
    '''
    # read data
    h, w = data.shape
    p_h,p_w = patch_size
    s_h,s_w = stride

    patches = []
    num = q*single_patches_num
    

    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        data_scaled = cv2.resize(data, (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
        for i in range(0, h_scaled-p_h+1, s_h):
            for j in range(0, w_scaled-p_w+1, s_w):
                x = data_scaled[i:i+p_h, j:j+p_w]
                
                if sum(sum(x)) != 0 and x.std() > 1e-5 and x.shape==patch_size:
                    num += 1
                    patch_num += 1
                    patches.append(x)
                    if verbose:
                        progress_bar(num,total_patches_num,patch_num,file,file_list)

                    if patch_num>=train_data_num:
                        return patches,patch_num

                    for k in range(0,aug_times):
                        x_aug = data_aug(x, mode=np.random.randint(0,8))
                        num += 1
                        patch_num += 1
                        patches.append(x_aug)
                        if verbose:
                            progress_bar(num,total_patches_num,patch_num,file,file_list)

                        if patch_num>=train_data_num:
                            return patches,patch_num
                elif verbose:
                    num = num+1+aug_times
                    progress_bar(num,total_patches_num,patch_num,file,file_list)

       
    return patches,patch_num

def datagenerator(data_dir,patch_size = (128,128),stride = (32,32), train_data_num = float('inf'),download=True,datasets = "Hess_VTI",aug_times=0,scales = [1],verbose=True,jump=1,agc=True):
    '''
    Args:
        data_dir : the path of the .segy file exit
        patch_size : the size the of patch
        stride : when get patches, the step size to slide on the data

        train_data_num: int or float('inf'),default=float('inf'),mean all the data will be used to Generate patches,
                        if you just need 3000 patches, you can set train_data_num=3000;
        download: bool; if you will download the dataset from the internet
        datasets : the num of the datasets will be download,if download = True
        aug_times : int, the time of the aug you will perform,used to increase the diversity of the samples,in each time,
                    Choose one operation at a time,eg:flip up and down、rotate 90 degree and flip up and down
        scales : list,The ratio of the data being scaled . default = [1],Not scaled by default.
        verbose: bool, Whether to output the generate situation of the patches

        jump : default=1, mean that read every shot data; when jump>=2, mean that don`t read the shot one by one
                instead of with a certain interval 

        agc : if use the agc of the data
    '''
    if download:
        if datasets>0: 
            Download_data(data_dir,datasets = datasets)
        else:
            print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")
            print("Please input the num of the dataset to download ")
            print("=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>")

    print('=> Generating patch samples')
    file_list = glob.glob(data_dir+'/*.segy')+glob.glob(data_dir+'/*.sgy')  # get name list of all .segy and .sgy files

    # initrialize
    all_patches=[]
    # generate patches
    patch_num = 0 
    for i in range(len(file_list)):
        with segyio.open(file_list[i],'r',ignore_geometry=True) as f:
            f.mmap()
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)#number of trace, The sourceX under the same shot is the same character.
            shot_num = len(set(sourceX))#shot number 
            len_shot = trace_num//shot_num #The length of the data in each shot data
            '''
            The data of each shot is read separately
            The default is that the data dimensions collected by all shots in the file are the same.
            Jump=1, which means that the data of all shots in the file is read by default. 
            When jump=2, it means that every other shot reads data.
            '''
            q = -1
            for j in range(0,shot_num,jump):
                data = np.asarray([np.copy(x) for x in f.trace[j*len_shot:(j+1)*len_shot]]).T
                q += 1
                if agc:
                    data = gain(data,0.004,'agc',0.05,1)
                else:
                    data = data/data.max()

                # Number of shots used to generate the patch
                select_shot_num = len(list(range(0,shot_num,jump)))

                h, w = data.shape
                p_h,p_w = patch_size
                s_h,s_w = stride
                single_patches_num = int(_compute_total_patches(h, w, p_h, p_w,s_h,s_w,aug_times,scales,max_patches=None))

                if verbose:
                    total_patches_num =  single_patches_num*select_shot_num                  
                    patches,patch_num = gen_patches(data,patch_size,stride,i,len(file_list),total_patches_num,train_data_num,patch_num,aug_times,scales,q,single_patches_num,verbose)
                else:
                    patches,patch_num = gen_patches(data,patch_size,stride,i,len(file_list),train_data_num = train_data_num,patch_num=patch_num,aug_times=aug_times,scales=scales,q = q,single_patches_num=single_patches_num)

                for patch in patches:
                    all_patches.append(patch)
                    if len(all_patches) >= train_data_num:
                        f.close()
                        if verbose:
                            print(' ')
                        all_patches = np.expand_dims(all_patches, axis=3)
                        print(str(len(all_patches))+' '+'training data finished')
                        return all_patches

            if verbose:
                print(' ')
            f.close()

    all_patches = np.expand_dims(all_patches, axis=3)

    #When the number of generated patches is an integer multiple of batch, run the following two lines of code.
#    discard_n = len(all_patches)-len(all_patches)//batch_size*batch_size   
#    all_patches = np.delete(all_patches, range(discard_n), axis=0)
    print(str(len(all_patches))+' '+'training data finished')

    return all_patches

if __name__ == '__main__':
    '''
    root (string): the .segy file exists or will be saved to if download is set to True.
    '''
#    torch.set_default_dtype(torch.float32) 
    root = 'data/test'
    train_data  = datagenerator(data_dir = root,patch_size = (128,128),stride = (32,32),train_data_num =1000,download=False,datasets =0,aug_times=9,scales = [1,0.9,0.8],verbose=False,jump=80,agc=False)
    train_data = train_data.astype(np.float64)
    torch.set_default_dtype(torch.float64)
    #just show some data sample form train_data
    xs = torch.from_numpy(train_data.transpose((0, 3, 1, 2)))
    DDataset = DenoisingDataset(xs,50)
#    DDataset = DownsamplingDataset(xs,4,regular = True)
    patch_show(DDataset,save=True,root = root) # show and save the 4 samples data
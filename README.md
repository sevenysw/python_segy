# Seismic data sample generation
Based on pytorch 

By Jing Wang and Siwei Yu (siweiyu@hit.edu.cn)

Center of Geophysics, Harbin Insititute of Technology, Harbin, China

If you find this toolbox useful, please cite the following paper (accepted by Geophysics):

Deep learning for denoising (https://arxiv.org/abs/1810.11614)

Note that the results from examples of this toolbox are not identical to those in the paper. The training set, test set, programming language are different. 

## Introduction
- This code is used to generate sample data from .segy seismic data  for deep learning based on pytorch.
- It can be used for denoising or interpolation for seismic data.
- This code is modified from [KaiZhang](https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch).
## Prerequisites
- Python3 with dependencies: scipy, numpy, h5py, glob,
[pytorch](https://github.com/pytorch/pytorch) and [segyio](https://github.com/equinor/segyio)
## Datasets
- you own **.segy** or **.sgy** seismic data or you can download some **.segy** or **.sgy** data online by the code we provide
- the model we provided is trained with [Model94_shots](http://s3.amazonaws.com/open.source.geoscience/open_data/bpmodel94/Model94_shots.segy.gz) and [7m_shots_0201_0329](http://s3.amazonaws.com/open.source.geoscience/open_data/bpstatics94/7m_shots_0201_0329.segy.gz) dataset (mode: DNCNN)
## Generating training data
### 


	from get_patch import*  
	from gain import * 
	# original data generates patch
	train_data = datagenerator(data_dir,patch_size = (128,128),stride = (32,32), train_data_num = float('inf'), download=False,datasets=[],aug_times=0,scales = [1],verbose=True,jump=1,agc=True)
	
	train_data = train_data.astype(np.float64)
	xs = torch.from_numpy(train_data.transpose((0, 3, 1, 2)))
    
	# add noise
    DDataset = DenoisingDataset(xs,25)

	'''
	#random downsampling，rate : the sampling rate
	DDataset = DownsamplingDataset(xs,rate = 0.7,regular = False)
	#sampling regularly, rate ： sampling interval
	DDataset = DownsamplingDataset(xs,rate = 2,regular = True)
	'''


Parameters in **datagenerator** :

    data_dir      : the path of the .segy file exit or you want to download in
    patch_size    : the size the of patch
    stride        : when get patches, the step size to slide on the data
    train_data_num: int or float('inf'),default=float('inf'),mean all the data will be used to Generate patches,
                    if you just need 3000 patches, you can set train_data_num=3000;

    download(bool): whether you will download the dataset from the internet,and we provide 7 inline datasets,the order is
				    1. http://s3.amazonaws.com/open.source.geoscience/open_data/bpmodel94/Model94_shots.segy.gz
                    2. http://s3.amazonaws.com/open.source.geoscience/open_data/bpstatics94/7m_shots_0201_0329.segy.gz
					3. https://s3.amazonaws.com/open.source.geoscience/open_data/bp2.5d1997/1997_2.5D_shots.segy.gz
					4. http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0001_0200.segy.gz
					5. http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part1.sgy.gz
					6. https://s3.amazonaws.com/open.source.geoscience/open_data/hessvti/timodel_shot_data_II_shot001-320.segy.gz
					7. http://s3.amazonaws.com/open.source.geoscience/open_data/Mobil_Avo_Viking_Graben_Line_12/seismic.segy

    datasets(int) : the number of the datasets will be download in the datasets we provide if download = True,
					 e.g:dataset=2,it mean that you will download the 1. http://s3.amazonaws.com/open.source.geoscience/open_data/bpmodel94/Model94_shots.segy.gz 
					 and 2. https://s3.amazonaws.com/open.source.geoscience/open_data/bp2.5d1997/1997_2.5D_shots.segy.gz two datasets.

    aug_times(int) : the time of the aug you will perform,used to increase the diversity of the samples,in each time,
                     Choose one operation at a time,eg:flip up and down、rotate 90 degree and flip up and down
    scales(list)   : The ratio of the data being scaled. default = [1], no scale by default.
    verbose(bool)  : Whether to output the generate situation of the patches

    jump(int)      : default=1, mean that read shot one by one; when jump>=2, mean that don`t read the shot one by one
                     instead of with a certain interval,such as: jump=3,you will use the 1、4、7... shot data

    agc(bool)      : if use the agc(Normalize each trace by amplitude) of the data

- **Note** : the parameters "jump" is only available when the dimensions of each shot data are the same. And we provide a small .segy data in ‘data/test’ to test the "datagenerator" function or you can just run `python get_patch.py` to test and look at some of the data sets that are being visualized. Just like:

![](https://wx1.sinaimg.cn/mw1024/006ceorLly1g3209srasvj30dz03itb0.jpg)
![](https://wx4.sinaimg.cn/mw1024/006ceorLly1g3209str92j30dw03igpc.jpg)
![](https://wx2.sinaimg.cn/mw1024/006ceorLly1g3209sozzwj30dz03iq59.jpg)
![](https://wx2.sinaimg.cn/mw1024/006ceorLly1g3209sr780j30dz03igoc.jpg)

## Training
	python main_train_denoise.py --data_dir data/train
	python main_train_inter.py --data_dir data/train

(Note: we suppose you have put the "segy" files in the "data/train" folder. If not, please use --download True --datasets 2 (2 means you want to use 2 datasets in the default library). Sometimes the network is not stable and the datasets cannot be downloaded. We provide a baiduyun link for some datasets here, link：https://pan.baidu.com/s/1YBO8-GOvk6JJGQZSKBdgJg)

## Test
	python main_test_denoise.py --data_dir data/test --sigma 50
![](https://wx3.sinaimg.cn/mw1024/006ceorLly1g31rqu5c7zj316y0bvng0.jpg)

	python main_test_inter.py --data_dir data/test --rate 2
![](https://wx4.sinaimg.cn/mw1024/006ceorLly1g31rqtq162j316w0c7aqa.jpg)


## Future work
- For more tasks: salt body classification、wave equation inversion and test for field data
- Parallel computing
- Support for matconvnet

	

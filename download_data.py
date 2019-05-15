# -*- coding: UTF-8 -*-

import urllib3
import os
import shutil
import gzip
import sys
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def Download_data(root,datasets=3):
    '''
    download the .segy file from the internet
    Args:
        root: the .segy file exists or will be saved to if download is set to True.
        datasets : name of the dataset if download is set to True.

    '''
    download_list=[
        "http://s3.amazonaws.com/open.source.geoscience/open_data/bpmodel94/Model94_shots.segy.gz",
        "http://s3.amazonaws.com/open.source.geoscience/open_data/bpstatics94/7m_shots_0201_0329.segy.gz",
        "https://s3.amazonaws.com/open.source.geoscience/open_data/bp2.5d1997/1997_2.5D_shots.segy.gz",
        "http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/shots0001_0200.segy.gz",
        "http://s3.amazonaws.com/open.source.geoscience/open_data/bptti2007/Anisotropic_FD_Model_Shots_part1.sgy.gz",
        "https://s3.amazonaws.com/open.source.geoscience/open_data/hessvti/timodel_shot_data_II_shot001-320.segy.gz",
        "http://s3.amazonaws.com/open.source.geoscience/open_data/Mobil_Avo_Viking_Graben_Line_12/seismic.segy"
            ]    

    for x in range(int(datasets)):
        url = download_list[x]        
        gz_filename = url.split("/")[-1]
        
        filename = gz_filename.replace(".gz","")
        
        gz_file_path = os.path.join(root, gz_filename)
        
        file_path = os.path.join(root, filename)
        #download and unzip
        if not os.path.exists(file_path):
            if not os.path.exists(gz_file_path):
                print('[%d/%d] downloading %s to %s'
                      % (x+1,int(datasets),download_list[x-1],file_path))
                
                r = requests.get(url, stream=True, verify=False)
                total_size = int(r.headers['Content-Length'])
                temp_size = 0
                with open(gz_file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            temp_size += len(chunk)
                            f.write(chunk)
                            f.flush()
                            #############downloading progress###############
                            done = int(50 * temp_size / total_size)
                            sys.stdout.write("\r[%s%s] %d%%" % ('#' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                            sys.stdout.flush()
                print()                            
                    
                with gzip.open(gz_file_path, 'rb') as read, open(file_path, 'wb') as write:
                    shutil.copyfileobj(read, write)
            else:

                print('%s already exists' % (filename))
                with gzip.open(gz_file_path, 'rb') as read, open(file_path, 'wb') as write:
                    shutil.copyfileobj(read, write)
        else:
            print('%s already exists' % (filename))
    print("download finished")

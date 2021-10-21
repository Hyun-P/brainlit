from cloudvolume import CloudVolume
import urllib.request
from skimage import io 
from joblib import Parallel, delayed
import multiprocessing
import os
import warnings
from tqdm import tqdm
import numpy as np

ncpu = 12
cis_path = "/data/jacsstorage/samples/tathey/bil1"
files_dir = '/data/jacsstorage/samples/tathey/files_bay'
progress_file = "/cis/home/tathey/progress.txt"

with open(progress_file) as f:
    for line in f:
        pass
    last_line = line
z_progress = int(last_line)

print(f"Starting at {z_progress}")

warnings.filterwarnings("ignore")


for z_start in tqdm(range(18, 10578, 165)):
    chunk = np.zeros((16000, 24000, 165), dtype='uint16')
    
    #download
    for z in range(z_start, z_start+165):
        filepath = '/data/tathey1/bil/files_bay/' + str(z) + '.tif'
        url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_373641-18462/CH1/18462_' + str(z).zfill(5) + '_CH1.tif'
        urllib.request.urlretrieve(url, filepath)
        # r = urllib.request.urlopen(url)
        # with open(filepath,'wb') as f:
        #     f.write(r.read())
        im = io.imread(filepath)
        print(im.shape)
        raise ValueError()


    for f in os.listdir(files_dir):
        os.remove(os.path.join(files_dir, f))
    break 


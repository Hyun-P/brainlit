from cloudvolume import CloudVolume
import urllib.request
from skimage import io 
from joblib import Parallel, delayed
import multiprocessing
import os
import warnings
from tqdm import tqdm
import numpy as np
import subprocess

ncpu = 12
cis_path = "/data/jacsstorage/samples/tathey/bil1/"
files_dir = '/data/jacsstorage/samples/tathey/files_bay/'
progress_file = "/cis/home/tathey/progress.txt"

def make_octree(dirs, levels_left):
    new_paths = []
    if levels_left > 0:
        for dir in dirs:
            for i in range(1,9):
                new_path = os.path.join(dir, str(i))
                new_paths.append(new_path)
                print(new_path)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)

        make_octree(new_paths, levels_left-1)
make_octree([cis_path], 6)



with open(progress_file) as f:
    for line in f:
        pass
    last_line = line
z_progress = int(last_line)

print(f"Starting at {z_progress}")

warnings.filterwarnings("ignore")

def compute_path(corner_coord):
    path = ""
    size = [16000, 24000, 10560]
    for level in range(6):
        size = np.divide(size, 2)
        comparison = np.greater_equal(corner_coord, size)
        folder = np.dot([2,1,4], comparison) + 1
        path = path + str(folder) + "/"
        corner_coord = np.subtract(corner_coord, np.multiply(comparison, size))
    z = int(corner_coord[2])
    return path, z

def download_z(z):
    cis_path = "/data/jacsstorage/samples/tathey/bil1/"
    files_dir = '/data/jacsstorage/samples/tathey/files_bay/'

    filepath = files_dir + str(z) + '.tif'
    url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_373641-18462/CH1/18462_' + str(z).zfill(5) + '_CH1.tif'
    subprocess.run(["wget", "-O", filepath, url])
    
    im = np.expand_dims(io.imread(filepath), axis=2)
    im = im[2000:18000,4000:28000,0]
    for i in range(0, im.shape[0], 250):
        for j in range(0, im.shape[1], 375):
            im_chunk = im[i:i+250, j:j+375]
            path, z = compute_path([i,j,z])
            out_path = cis_path + path + str(z) + ".tif"
            io.imsave(out_path, im_chunk)


for z_start in tqdm(range(18, 10578, ncpu)):

    with open(progress_file, 'a') as f:
        f.write('\n')
        f.write(f'{z_start}')

    Parallel(n_jobs=ncpu)(delayed(download_z)(z) for z in range(z_start, z_start+ncpu))

    for f in os.listdir(files_dir):
        os.remove(os.path.join(files_dir, f))
    break 


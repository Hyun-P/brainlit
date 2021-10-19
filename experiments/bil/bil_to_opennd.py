from cloudvolume import CloudVolume
import urllib.request
from skimage import io 
from joblib import Parallel, delayed
import multiprocessing
import os
import warnings

ncpu = 12
precomputed_path = "s3://open-neurodata/brainlit/bil1"
files_dir = '/data/tathey1/bil/files_bay/'
progress_file = "/home/tathey1/progress.txt"
vol = CloudVolume(precomputed_path)
print(f"cv shape: {vol.shape}")

with open(progress_file) as f:
    for line in f:
        pass
    last_line = line
z_progress = int(last_line)

print(f"Starting at {}")

warnings.filterwarnings("ignore")

def upload_z(z):
    filepath = '/data/tathey1/bil/files_bay/' + str(z) + '.tif'

    z2 = z+1
    url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_373641-18462/CH1/18462_' + z2.toString("D5") + '_CH1.tif'

    r = urllib.request.urlopen(url)
    with open(filepath,'wb') as f:
        f.write(r.read())

    im = io.imread(filepath)
    vol[:,:,z] = im.T

for z_start in range(z_progress, vol.shape[2], ncpu):

    with open(progress_file, 'a') as f:
        f.write('\n')
        f.write(f'{z_start}')

    Parallel(n_jobs=ncpu)(delayed(upload_z)(z) for z in range(z_start, z_start+ncpu))

    for f in os.listdir(files_dir):
        os.remove(os.path.join(files_dir, f))


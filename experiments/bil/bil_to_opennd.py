from cloudvolume import CloudVolume
import urllib.request
from skimage import io 
from joblib import Parallel, delayed
import multiprocessing
import os

ncpu = 12
precomputed_path = "s3://open-neurodata/brainlit/bil1"
files_dir = '/data/tathey1/bil/files_bay/'
vol = CloudVolume(precomputed_path)
print(f"cv shape: {vol.shape}")

def upload_z(z):
    filepath = '/data/tathey1/bil/files_bay/' + str(z) + '.tif'

    z2 = z+1
    url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_373641-18462/CH1/18462_' + z2.toString("D5") + '_CH1.tif'

    r = urllib.request.urlopen(url)
    with open(filepath,'wb') as f:
        f.write(r.read())

    im = io.imread(filepath)
    vol[:,:,z] = im.T

for z_start in range(vol.shape[2], ncpu):
    Parallel(n_jobs=ncpu)(delayed(upload_z)(z) for z in range(z_start, z_start+ncpu))
    
    for f in os.listdir(files_dir):
        os.remove(os.path.join(files_dir, f))

import chunk
import numpy as np
import fsspec, requests
from bs4 import BeautifulSoup
from skimage import io
import PIL.Image
import zarr
from tqdm import tqdm
from brainlit.algorithms.generate_fragments.state_generation import state_generation
from skimage import measure
from scipy import stats

PIL.Image.MAX_IMAGE_PIXELS = 1056323868


def download_image():
    def getFilesHttp(url: str,ext: str) -> list:
        def listFD(url, ext=''):
            page = requests.get(url).text
            # print(page)
            soup = BeautifulSoup(page, 'html.parser')
            return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
        
        files = []
        for file in listFD(url, ext):
            files.append(file)
            
        return files

    def getImage(fileObj):
        with fileObj as f:
            #print('Reading {} \n'.format(f))
            return io.imread(f)

    url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_362188-191815/CH1_0.35_100um/'
    files = sorted(getFilesHttp(url, "tif"))
    files = [fsspec.open(x,'rb') for x in files]

    image = getImage(files[0])
    #image_zarr = zarr.zeros((len(files), image.shape[0], image.shape[1]), chunks=(1,1000,1000), dtype=image.dtype)
    image_zarr = zarr.open("/data/tathey1/bil/image.zarr", mode='w', shape=(len(files), image.shape[0], image.shape[1]), chunks=(1,1000, 1000), dtype=image.dtype)

    for i, fileObj in enumerate(tqdm(files)):
        image = getImage(fileObj)
        image_zarr[i,:,:] = image

    #zarr.save("data/tathey1/bil/image.zarr", image_zarr)


def viterbrain():
    sg = state_generation(
        image_path="/data/tathey1/bil/image.zarr",
        ilastik_program_path="/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh",
        ilastik_project_path="/data/tathey1/bil/ilastik/bil_slice.ilp",
        chunk_size=[5,500,500],
        soma_coords=[],
        resolution = [100, 0.35, 0.35],
        parallel=24,
        prob_path="/data/tathey1/bil/image_probs.zarr",
        fragment_path="/data/tathey1/bil/image_labels.zarr",
        tiered_path="/data/tathey1/bil/image_tiered.zarr",
        states_path ="/data/tathey1/bil/image_nx.pickle"
    )

    # sg.predict(
    #     data_bin="/data/tathey1/bil/files_bay/",
    #     pos_class = 0
    # )

    #sg.compute_frags()
    #sg.compute_soma_lbls()
    # sg.compute_image_tiered()
    #sg.compute_states()
    sg.compute_edge_weights()


labs = zarr.open("/data/tathey1/bil/image_labels.zarr")
spacing = (100,100,100)
func = stats.mode

new_size = [np.ceil(shap/space) for shap,space in zip(labs.shape, spacing)]
new_chunks = [int(np.amax([np.floor(chunk/space), 1])) for chunk,space in zip(labs.chunks, spacing)]
labs_ds = zarr.open("/data/tathey1/bil/image_labels_ds.zarr", "w", shape=new_size, chunks=new_chunks, dtype="i4")
print(f"Writing {labs_ds} with shape {labs_ds.shape}, chunks {labs_ds.chunks}, and dtype {labs_ds.dtype}")


for ix,x1 in enumerate(tqdm(range(0, labs.shape[0], spacing[0]), desc="x")):
    x2 = np.amin([labs.shape[0], x1 + spacing[0]])
    for iy, y1 in enumerate(tqdm(range(0, labs.shape[1], spacing[1]), desc="y", leave=False)):
        y2 = np.amin([labs.shape[1], y1 + spacing[1]])
        for iz, z1 in enumerate(range(0, labs.shape[2], spacing[2])):
            z2 = np.amin([labs.shape[2], z1 + spacing[2]])
            im = labs[x1:x2,y1:y2,z1:z2]
            val = func(im, axis=None)[0][0]
            print(val)
            labs_ds[ix,iy,iz] = val


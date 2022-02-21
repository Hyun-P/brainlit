import chunk
import fsspec, requests
from bs4 import BeautifulSoup
from skimage import io
import PIL.Image
import zarr
from tqdm import tqdm
from brainlit.algorithms.generate_fragments.state_generation import state_generation
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
        ilastik_project_path="/data/tathey1/bil/ilaztik/bil_slice.ilp",
        chunk_size=[5,1000,1000],
        soma_coords=[],
        parallel=16,
    )

    sg.compute_frags()
    sg.compute_soma_lbls()
    sg.compute_image_tiered()
    sg.compute_states()
    sg.compute_edge_weights()

viterbrain()



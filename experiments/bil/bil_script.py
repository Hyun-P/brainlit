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
from joblib import Parallel, delayed
import pickle

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
        ilastik_project_path="/data/tathey1/bil/ilastik_2d/bil_slice_2d.ilp",
        chunk_size=[1,500,500],
        soma_coords=[],
        resolution = [100, 0.35, 0.35],
        parallel=8,
        prob_path="/data/tathey1/bil/image_probs.zarr",
        fragment_path="/data/tathey1/bil/image_labels.zarr",
        tiered_path="/data/tathey1/bil/image_tiered.zarr",
        states_path ="/data/tathey1/bil/image_nx.pickle"
    )

    # sg.predict(
    #     data_bin="/data/tathey1/bil/files_bay/",
    #     pos_class = 0
    # )

    # sg.compute_frags(threshold=0.25)
    # sg.compute_soma_lbls()
    # sg.compute_image_tiered()
    # sg.compute_states()
    # sg.compute_edge_weights()


print("loading viterbrain object")
with open("/data/tathey1/bil/image_viterbrain_geomoonly.pickle", "rb") as handle:
    viterbrain = pickle.load(handle)

print('running computatino')
viterbrain.compute_all_costs_int()
print(f"# Edges: {viterbrain.nxGraph.number_of_edges()}")

with open("/data/tathey1/bil/image_viterbrain.pickle", "wb") as handle:
    pickle.dump(viterbrain, handle)

# Downsample


# labs = zarr.open("/data/tathey1/bil/image_labels.zarr")
# spacing = (1,10,10)
# func = stats.mode
# parallel = 8

# new_size = [np.ceil(shap/space) for shap,space in zip(labs.shape, spacing)]
# new_chunks = [int(np.amax([np.floor(chunk/space), 1])) for chunk,space in zip(labs.chunks, spacing)]
# labs_ds = zarr.open("/data/tathey1/bil/image_labels_ds.zarr", "w", shape=new_size, chunks=new_chunks, dtype="i4")
# print(f"Writing {labs_ds} with shape {labs_ds.shape}, chunks {labs_ds.chunks}, and dtype {labs_ds.dtype}")



# def _get_frag_specifications(image, chunk_size):
#     num_chunks_per_block = 10 ** 9
#     specifications = []

#     for ix, x in enumerate(np.arange(0, image.shape[0], chunk_size[0])):
#         x2 = np.amin([x + chunk_size[0], image.shape[0]])
#         for iy, y in enumerate(np.arange(0, image.shape[1], chunk_size[1])):
#             y2 = np.amin([y + chunk_size[1], image.shape[1]])
#             for iz, z in enumerate(np.arange(0, image.shape[2], chunk_size[2])):
#                 z2 = np.amin([z + chunk_size[2], image.shape[2]])

#                 specifications.append(
#                     {
#                         "corner1": [x, y, z],
#                         "corner2": [x2, y2, z2],
#                         "idx": [ix, iy, iz]
#                         }
#                     )
#         specifications = [
#             specifications[x : x + num_chunks_per_block]
#             for x in range(0, len(specifications), num_chunks_per_block)
#         ]

#         return specifications

# def downsample_block(corner1, corner2, idx):
#     im = labs[corner1[0]:corner2[0],corner1[1]:corner2[1],corner1[2]:corner2[2]]
#     val = func(im, axis=None)[0][0]

#     return (val, idx)



# specification_blocks = _get_frag_specifications(labs, spacing)

# for i, specifications in enumerate(tqdm(specification_blocks, desc="Downsampling")):
#     results = Parallel(n_jobs=parallel)(
#         delayed(downsample_block)(
#             specification["corner1"],
#             specification["corner2"],
#             specification["idx"],
#         )
#         for specification in tqdm(
#             specifications,
#             desc=f"Computing labels {i}: {specifications[0]}, {specifications[-1]}",
#             disable = False,
#             leave = False
#         )
#     )

#     for result in tqdm(
#         results,
#         desc=f"Writing block {i}: {specifications[0]}, {specifications[-1]}",
#         leave = False
#     ):
#         val, idx = result
#         labs_ds[idx[0], idx[1], idx[2]] = val
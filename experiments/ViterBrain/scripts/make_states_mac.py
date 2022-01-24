from brainlit.algorithms.generate_fragments import state_generation
import time

t1 = time.perf_counter()
sg = state_generation(
    "/Users/thomasathey/Documents/mimlab/mouselight/input/images/big/250.zarr",
    "/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/run_ilastik.sh",
    "/Users/thomasathey/Documents/mimlab/mouselight/octopus_experiment/octopus_exp.ilp",
    chunk_size=[300, 300, 300],
    parallel=12)
print(f"create object in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.predict("/Users/thomasathey/Documents/mimlab/mouselight/input/images/big/data_bin/")
print(f"computed ilastik predictions in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_frags()
print(f"computed fragments in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_image_tiered()
print(f"computed tiered image in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_soma_lbls()
print(f"computed soma labels in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_states()
print(f"computed states in {time.perf_counter()-t1} seconds")


t1 = time.perf_counter()
sg.compute_edge_weights()
print(f"computed edge weights in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_bfs()
print(f"computed bfs tree in {time.perf_counter()-t1} seconds")
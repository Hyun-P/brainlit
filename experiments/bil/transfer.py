import igneous.task_creation as tc
from taskqueue import LocalTaskQueue

src_layer_path = "s3://open-neurodata/brainlit/bil1/"
dest_layer_path = "s3://open-neurodata/brainlit/bil1_resample/"

tq = LocalTaskQueue(parallel=8)

tasks = tc.create_transfer_tasks(
  src_layer_path, dest_layer_path, 
  chunk_size=(64,64,64)
)

tq.insert(tasks)
tq.execute()
# only parameters to be in this file

img_sz = 32

image_dims = (3, img_sz, img_sz) # c, h, w
starting_filters = 64

time_steps = 1000

device = "cuda"
batch_size = 24
epochs = 10000
lr = 1e-3
dataset_path = "/mnt/d/work/datasets/bikes/bikes_clean"

model_save_dir = "./models"
img_save_dir = "./samples"
metrics_save_dir = "./metrics"

use_ddim_sampling = False


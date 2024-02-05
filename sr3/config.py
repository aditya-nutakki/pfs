# only parameters to be in this file

img_sz = 128

image_dims = (3, img_sz, img_sz) # c, h, w
starting_filters = 64

time_steps = 2000

device = "cuda"
batch_size = 24
epochs = 10000
lr = 5e-4

hr_sz, lr_sz = 128, 16

dataset_path = "/mnt/d/work/datasets/bikes/bikes_clean"

exp_name = "celeba"
model_save_dir = f"./{exp_name}_models"
img_save_dir = f"./{exp_name}_samples"
metrics_save_dir = f"./{exp_name}_metrics"

use_ddim_sampling = False


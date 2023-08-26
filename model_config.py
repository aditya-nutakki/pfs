from os import listdir

batch_size = 8
epochs = 30
mode = "train" # can be "infer" to put model in inference mode or "train" to train model
input_dims = (3, 224, 224) # make sure to modify the beginning and the last layers of any model in train_module to get the correct shapes; this is the expected size for both the teacher and student model

# train_path, test_path = "/mnt/d/work/datasets/rice_image_dataset/train", "/mnt/d/work/datasets/rice_image_dataset/test"
train_path, test_path = "", ""
device = "cuda" # or "cpu"
save_dir, model_name = "./models", "base" # .pt extension will be added, 'None' if you dont want to save 
nc = len(listdir(train_path)) # number of classes
log_path = f"./logs_{model_name}.txt"
model_path = "./models/base_epoch1.pt"
base_model_path = "./models/resnet50_epoch2.pt"
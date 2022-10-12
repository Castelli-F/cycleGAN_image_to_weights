import inspect
import time
import os

# MAIN ROOT
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir = os.path.dirname(os.path.dirname(current_dir)) + "/dataset/"
print("la current directory è:", current_dir)
print("la dataset directory è:", dataset_dir)
# LOSS PLOT DIRECTORY
PLOT_PATH = current_dir + "/LOSS"
# IMG DIRECTORY
IMG_DIR = dataset_dir + "cycle_dataset/"
# MODEL SAVE DIRECTORY
MODEL_GEN_G_PATH = current_dir + "/MODEL_GEN_G"
MODEL_GEN_F_PATH = current_dir + "/MODEL_GEN_F"
MODEL_DISC_X_PATH = current_dir + "/MODEL_DISC_X"
MODEL_DISC_Y_PATH = current_dir + "/MODEL_DISC_Y"

# RECONSTRUCTED IMAGES DIRECTORY
IMAGES = current_dir + "/RECONSTRUCTED IMAGES"

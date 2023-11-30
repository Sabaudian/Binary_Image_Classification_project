# import
import os

# -------------------------------------- #
# ---------- USEFUL CONSTANTS ---------- #
# -------------------------------------- #

# DATASET AND DATA LOCATION
DATASET_PATH = "dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")  # DATASET_PATH + "/train"
TEST_DIR = os.path.join(DATASET_PATH, "test")  # DATASET_PATH + "/test"
DATA_PATH = "data"

# IMAGE VALUE AND DATA
FILE_EXTENSION = ".jpg"
COLOR_MODE = "L"  # grayscale
IMG_WIDTH = 192
IMG_HEIGHT = 192
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

# FOR PLOT AND SIMILAR
PLOT_FOLDER = "plot"
CLASS_LIST = ["chihuahua", "muffin"]

# KERAS, HYPERPARAMETERS AND SIMILAR
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# KFOLD CROSS-VALIDATION AND SIMILAR
K_FOLD = 5

# import
import os

# ************************************** #
# ************** CONSTANTS ************* #
# ************************************** #

# DATASET, PATH AND SIMILAR
DATASET_PATH = "dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")  # DATASET_PATH + "/train"
TEST_DIR = os.path.join(DATASET_PATH, "test")  # DATASET_PATH + "/test"
DATA_PATH = "data"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # The project root
# Info. to download dataset from Kaggle
DATASET_ID = "samuelcortinhas/muffin-vs-chihuahua-image-classification"
# Google Drive URL
DRIVE_URL = "https://drive.google.com/drive/folders/1Cd3KdUZ77dV7lPmpnhDDf6lXKFgFOa1u?usp=share_link"


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

# KERAS AND SIMILAR
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# KFOLD CROSS-VALIDATION AND SIMILAR
KFOLD = 5

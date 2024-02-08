# import
import os

# ************************************** #
# ************** CONSTANTS ************* #
# ************************************** #

# PATHS AND SIMILAR
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # The project root
DATASET_PATH = "dataset"  # Path to dataset folder
TRAIN_DIR = os.path.join(DATASET_PATH, "train")  # Path to train set folder
TEST_DIR = os.path.join(DATASET_PATH, "test")   # Path to test set folder
DATA_PATH = "data"  # Path to data folder
MODELS_PATH = "models"  # Path to models folder
PLOT_FOLDER = "plot"  # Path to plot folder
DATASET_ID = "samuelcortinhas/muffin-vs-chihuahua-image-classification"  # Necessary for download dataset (Kaggle API)

# IMAGE VALUE AND DATA
FILE_EXTENSION = ".jpg"
IMG_WIDTH = 192
IMG_HEIGHT = 192
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

# PLOT AND SIMILAR
CLASS_LIST = ["chihuahua", "muffin"]

# KERAS AND SIMILAR
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# K-FOLD CROSS-VALIDATION
KFOLD = 5

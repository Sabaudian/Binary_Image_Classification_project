# -------------------------------------- #
# ---------- USEFUL CONSTANTS ---------- #
# -------------------------------------- #

# DATASET AND DATA LOCATION
DATASET_PATH = "dataset"
TRAIN_DIR = DATASET_PATH + "/train"
TEST_DIR = DATASET_PATH + "/test"
DATA_PATH = "data"
REPORT_PATH = "report"

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

# FOR NAMING FILE AND SIMILAR
DATASET_VISUAL_INFO = "dataset_visualization_plot"

# KERAS, HYPERPARAMETERS AND SIMILAR
BATCH_SIZE = 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# KFOLD CROSS-VALIDATION AND SIMILAR
K_FOLD = 5
# NN_MODEL_CHECKPOINT =
# MLP_MODEL_CHECKPOINT =
# CNN_MODEL_CHECKPOINT =


# NOTE PROGETTO:
# modelli da usare: vgg-16, resnet50, mobileNet (approfondire vit)

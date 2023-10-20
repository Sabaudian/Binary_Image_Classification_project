# -------------------------------------- #
# ---------- USEFUL CONSTANTS ---------- #
# -------------------------------------- #

# DATASET AND DATA LOCATION
DATASET_PATH = "dataset"
TRAIN_DIR = DATASET_PATH + "/train"
TEST_DIR = DATASET_PATH + "/test"
DATA_PATH = "data"

# IMAGE VALUE AND DATA
FILE_EXTENSION = ".jpg"
COLOR_MODE = "L"  # grayscale
IMG_WIDTH = 180
IMG_HEIGHT = 180
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# FOR PLOT AND SIMILAR
PLOT_FOLDER = "plot"

# FOR NAMING FILE AND SIMILAR
DATASET_VISUAL_INFO = "dataset_visualization_plot"

# KERAS, HYPERPARAMETERS AND SIMILAR
BATCH_SIZE = 32
INPUT_SHAPE = IMG_SIZE + (3,)


# NOTE PROGETTO:
# modelli da usare: vgg-16, resnet50, mobileNet (approfondire vit)

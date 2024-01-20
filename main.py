# My import
import constants as const

from utils.pre_processing import checking_dataset
from utils.general_functions import download_models_saves_from_drive

from classifiers import classification_and_evaluation

# Imported to get package version info.
import sklearn
import platform
import tensorflow

# import warnings
#
# warnings.filterwarnings("ignore")


# ************************ #
# ********* MAIN ********* #
# ************************ #

# Main class of the project
def main():
    """
    Main function.

    This function provides information about the versions of Python and relevant packages
    (Scikit-learn, Tensorflow), checks the dataset if required, and performs classification
    and evaluation of models.

    :return: None
    """
    # Python Packages Version info.
    print("\n> Version control")
    print("- Python version is: {}".format(platform.python_version()))
    print("- Scikit-learn version is: {}".format(sklearn.__version__))
    print("- Tensorflow version is: {}".format(tensorflow.__version__))
    print("______________________________________________________________________________")

    # Checking the dataset
    check_dataset = input("> Preprocessing: is it necessary to check the dataset? [Y/N]: ")
    if check_dataset.upper() == "Y":
        checking_dataset(dataset_path=const.DATASET_PATH,
                         train_dir_path=const.TRAIN_DIR,
                         test_dir_path=const.TEST_DIR,
                         show_plot=False, save_plot=True)
    print("______________________________________________________________________________")

    # Download the folder "models" from drive to speed up the process
    download_models_saves_from_drive(drive_url=const.DRIVE_URL, root_dir=const.PROJECT_ROOT)

    # Classification and Evaluation of the Models
    classification_and_evaluation(train_path=const.TRAIN_DIR, test_path=const.TEST_DIR,
                                  random_prediction=False, show_plot=False, save_plot=True)


if __name__ == '__main__':
    main()

# My import
import constants as const
from utils.pre_processing import checking_dataset
from classifiers import classification_and_evaluation
from utils.general_functions import download_models_save_from_drive

# Imported to get package version info.
import sklearn
import platform
import tensorflow

import warnings

warnings.filterwarnings("ignore")


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
    download_models_save_from_drive(drive_url=const.DRIVE_URL, root_dir=const.PROJECT_ROOT)

    # Classification and Evaluation of the Models
    classification_and_evaluation(train_path=const.TRAIN_DIR, test_path=const.TEST_DIR,
                                  show_plot=False, save_plot=True)


if __name__ == '__main__':
    main()

    # # My import
    # import classifiers
    # import models_evaluation as evaluate
    # import constants as const
    # import prepare_dataset as prepare
    #
    # # Load keras datasets
    # train_dataset, val_dataset, test_dataset = prepare.load_dataset(train_data_dir=const.TRAIN_DIR,
    #                                                                 test_data_dir=const.TEST_DIR)
    #
    # # Printing information about the datasets
    # print("\n> Class Names:"
    #       "\n\t- Class 0 = {}"
    #       "\n\t- Class 1 = {}".format(train_dataset.class_names[0], train_dataset.class_names[1]))
    #
    # # Scaling data
    # train_ds = prepare.data_normalization(tf_dataset=train_dataset, augment=True)
    # val_ds = prepare.data_normalization(tf_dataset=val_dataset, augment=False)
    # test_ds = prepare.data_normalization(tf_dataset=test_dataset, augment=False)
    #
    # plot_functions.plot_data_augmentation(train_ds=train_ds, data_augmentation=prepare.perform_data_augmentation(),
    #                                       show_on_screen=True, store_in_folder=True)
    #
    # # dataset into array
    # X_train, y_train = prepare.image_to_array(train_ds)
    # X_val, y_val = prepare.image_to_array(val_ds)
    # X_test, y_test = prepare.image_to_array(test_ds)
    #
    # # NN Model Tuning
    # cnn_model = classifiers.build_cnn_model
    # classifiers.tuning_hyperparameters(model=cnn_model, model_name="CNN",
    #                                    x_train=X_train, y_train=y_train,
    #                                    x_val=X_val, y_val=y_val)
    #
    # evaluate.evaluate_model(model=cnn_model, model_name="CNN",
    #                         x_test=X_test, y_test=y_test, show_plot=False, save_plot=True)

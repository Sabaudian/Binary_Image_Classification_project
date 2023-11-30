# My import
import constants as const
from utils.pre_processing import checking_dataset
from classifiers import classification_and_evaluation

# Imported to get package version info.
import sklearn
import platform
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


# ************************ #
# ********* MAIN ********* #
# ************************ #

# Main
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
    print("- Tensorflow version is: {}".format(tf.__version__))
    print("______________________________________________________________________________")

    # Checking the dataset
    check_dataset = input("> Preprocessing: Is it necessary to check the dataset? [Y/N]: ")
    if check_dataset.upper() == "Y":
        checking_dataset(dataset_path=const.DATASET_PATH,
                         train_dir_path=const.TRAIN_DIR,
                         test_dir_path=const.TEST_DIR,
                         show_plot=False, save_plot=True)
    print("______________________________________________________________________________")

    # Classification and Evaluation of the Models
    classification_and_evaluation(train_path=const.TRAIN_DIR, test_path=const.TEST_DIR, show_plot=False, save_plot=True)


# Main class of the project
if __name__ == '__main__':

    main()

    # # Python Packages Version info.
    # print("\n> Version control")
    # print("- Python version is: {}".format(platform.python_version()))
    # print("- Scikit-learn version is: {}".format(sklearn.__version__))
    # print("- Tensorflow version is: {}".format(tf.__version__))
    # print("______________________________________________________________________________")

    # # Checking the dataset
    # check_dataset = input("> Preprocessing: Is it necessary to check the dataset? [Y/N]: ")
    # if check_dataset.upper() == "Y":
    #     checking_dataset(dataset_path=const.DATASET_PATH,
    #                      train_dir_path=const.TRAIN_DIR,
    #                      test_dir_path=const.TEST_DIR,
    #                      show_plot=False, save_plot=False)
    # print("______________________________________________________________________________")

    # # Classification and Evaluation of the Models
    # classification_and_evaluation(train_path=const.TRAIN_DIR, test_path=const.TEST_DIR)

    # # Load keras datasets
    # train_dataset, val_dataset, test_dataset = prepare.load_dataset(train_data_dir=const.TRAIN_DIR,
    #                                                                 test_data_dir=const.TEST_DIR)
    #
    # # Printing information about the datasets
    # print("\n> Class Names:"
    #       "\n\t- classe 0 = {}"
    #       "\n\t- classe 1 = {}".format(train_dataset.class_names[0], train_dataset.class_names[1]) +
    #       "\n> Type specification:\n\t- {}\n".format(train_dataset.element_spec))
    #
    # # Visualize the dataset showing some images with corresponding labels
    # plot_functions.plot_view_dataset(train_ds=train_dataset, show_on_screen=False, store_in_folder=False)
    #
    # # Scaling data
    # train_ds = prepare.data_normalization(tf_dataset=train_dataset, augment=True)
    # val_ds = prepare.data_normalization(tf_dataset=val_dataset, augment=False)
    # test_ds = prepare.data_normalization(tf_dataset=test_dataset, augment=False)
    #
    # # Visualize the data_augmentation process effect
    # plot_functions.plot_data_augmentation(train_ds=train_dataset, data_augmentation=prepare.perform_data_augmentation(),
    #                                       show_on_screen=False, store_in_folder=False)
    #
    # # test con array
    # X_train, y_train = prepare.image_to_array(train_ds)
    # X_val, y_val = prepare.image_to_array(val_ds)
    # X_test, y_test = prepare.image_to_array(test_ds)

    # NN Model Tuning
    # nn_model = classifiers.build_nn_model
    # classifiers.tuning_hyperparameters(model=nn_model, model_name="NN",
    #                                    x_train=X_train, y_train=y_train,
    #                                    x_val=X_val, y_val=y_val)
    # NN KFold cross-validation
    # kfold_nn_model = classifiers.kfold_cross_validation(model_name="NN",
    #                                                     x_train=X_train, y_train=y_train,
    #                                                     x_val=X_val, y_val=y_val,
    #                                                     k_folds=const.K_FOLD)
    # # Evaluate NN model
    # evaluate.evaluate_model(model=kfold_nn_model, model_name="NN", x_test=X_test, y_test=y_test)
    #
    # # MLP Model Tuning
    # mlp_model = classifiers.build_mlp_model
    # classifiers.tuning_hyperparameters(model=mlp_model, model_name="MLP", x_train=X_train,
    #                                    y_train=y_train, x_val=X_val, y_val=y_val)
    # # MLP KFold cross-validation
    # kfold_mlp_model = classifiers.kfold_cross_validation(model_name="MLP", x_train=X_train,
    #                                                      y_train=y_train, x_val=X_val, y_val=y_val,
    #                                                      k_folds=const.K_FOLD)
    # # Evaluate MLP model
    # evaluate.evaluate_model(model=kfold_mlp_model, model_name="MLP", x_test=X_test, y_test=y_test)
    #
    # CNN model tuning
    # cnn_model = classifiers.build_cnn_model
    # classifiers.tuning_hyperparameters(model=cnn_model, model_name="CNN", x_train=X_train,
    #                                    y_train=y_train, x_val=X_val, y_val=y_val)
    #
    # # CNN KFold cross-validation
    # kfold_cnn_model = classifiers.kfold_cross_validation(model_name="CNN", x_train=X_train,
    #                                                      y_train=y_train, x_val=X_val, y_val=y_val,
    #                                                      k_folds=const.K_FOLD)
    # # Evaluate CNN model
    # evaluate.evaluate_model(model=kfold_cnn_model, model_name="CNN", x_test=X_test, y_test=y_test)
    #
    # # MobileNet model tuning
    # mobilenet_model = classifiers.build_mobilenet_model
    # # MobileNet Tuning hyperparameters
    # classifiers.tuning_hyperparameters(model=mobilenet_model, model_name="MobileNet",
    #                                    x_train=X_train, y_train=y_train,
    #                                    x_val=X_val, y_val=y_val)
    # # MobileNet KFold cross-validation
    # kfold_mobilenet_model = classifiers.kfold_cross_validation(model_name="MobileNet",
    #                                                            x_train=X_train, y_train=y_train,
    #                                                            x_val=X_val, y_val=y_val,
    #                                                            k_folds=const.K_FOLD)
    # # Evaluate MobileNet model
    # evaluate.evaluate_model(model=kfold_mobilenet_model, model_name="MobileNet", x_test=X_test, y_test=y_test)
    #
    # # VGG16 model tuning
    # vgg16_model = classifiers.build_vgg16_model
    # classifiers.tuning_hyperparameters(model=vgg16_model, model_name="VGG16", x_train=X_train,
    #                                    y_train=y_train, x_val=X_val, y_val=y_val)
    #
    # # # VGG16 KFold cross-validation
    # kfold_mobilenet_model = classifiers.kfold_cross_validation(model_name="VGG16",
    #                                                            x_train=X_train, y_train=y_train, x_val=X_val,
    #                                                            y_val=y_val,
    #                                                            k_folds=const.K_FOLD)
    # # Evaluate VGG16 model
    # evaluate.evaluate_model(model=kfold_mobilenet_model, model_name="VGG16", x_test=X_test, y_test=y_test)

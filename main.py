# My import
import keras
import classifiers
import plot_functions
import constants as const
import prepare_dataset as prepare
import models_evaluation as evaluate
import utils.general_functions as general
import utils.pre_processing as pre_processing

# Imported to get package version info.
import sys
import sklearn
import tensorflow as tf

# ************************ #
# ********* MAIN ********* #
# ************************ #

# Main class of the project
if __name__ == '__main__':
    """
    EXPERIMENTAL PROJECTS - NEURAL NETWORK
    Use Keras to train a neural network for the binary classification of muffins and Chihuahuas 
    based on images from this dataset.
    Images must be transformed from JPG to RGB (or Grayscale) pixel values and scaled down. 

    The student is asked to:
        - experiment with different network architectures (at least 3) and training hyperparameters,
        - use 5-fold cross validation to compute your risk estimates,
        - thoroughly discuss the obtained results, documenting the influence of the choice of
          the network architecture and the tuning of the hyperparameters on the final
          cross-validated risk estimate.

    While the training loss can be chosen freely, the reported cross-validated estimates must be 
    computed according to the zero-one loss.
    """
    # Python Packages Version info.
    print("\n> Version control")
    print("- Python version is: {}".format(sys.version))
    print("- Scikit-learn version is: {}".format(sklearn.__version__))
    print("- Tensorflow version is: {}".format(tf.__version__))
    print("_________________________________________________________________")

    # # Checking the dataset
    # check_dataset = input("> START TO CHECK THE DATASET? [Y/N]: ")
    # if check_dataset.upper() == "Y":
    #     pre_processing.checking_dataset(dataset_path=const.DATASET_PATH,
    #                                     train_dir_path=const.TRAIN_DIR,
    #                                     test_dir_path=const.TEST_DIR,
    #                                     show=False,
    #                                     save=False)

    # Load keras datasets
    train_dataset, val_dataset, test_dataset = prepare.load_dataset(train_data_dir=const.TRAIN_DIR,
                                                                    test_data_dir=const.TEST_DIR)

    # Printing information about the datasets
    print("\n> Class Names:"
          "\n\t- classe 0 = {}"
          "\n\t- classe 1 = {}".format(train_dataset.class_names[0], train_dataset.class_names[1]) +
          "\n> Type specification:\n\t- {}\n".format(train_dataset.element_spec))

    # Visualize the dataset with corresponding labels
    plot_functions.plot_view_dataset(train_ds=train_dataset, show_on_screen=False, store_in_folder=False)

    # Scaling data
    train_ds = prepare.data_normalization(tf_dataset=train_dataset, augment=True)
    val_ds = prepare.data_normalization(tf_dataset=val_dataset, augment=False)
    test_ds = prepare.data_normalization(tf_dataset=test_dataset, augment=False)

    # Visualize the data_augmentation process effect
    plot_functions.plot_data_augmentation(train_ds=train_dataset, data_augmentation=prepare.data_augmentation,
                                          show_on_screen=False, store_in_folder=True)

    # test con array
    X_train, y_train = prepare.image_to_array(train_ds)
    X_val, y_val = prepare.image_to_array(val_ds)
    X_test, y_test = prepare.image_to_array(test_ds)

    print(X_train.shape)

    # Neural Network model tuning
    nn_model = classifiers.build_nn_model
    tuned_nn_model = classifiers.tuning_hyperparameters(model=nn_model, model_name="NN", x_train=X_train,
                                                        y_train=y_train, x_val=X_val, y_val=y_val)
    # Evaluate NN model
    evaluate.evaluate_model(model=tuned_nn_model, model_name="NN",
                            x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # MLP Model Tuning
    # mlp_model = classifiers.build_mlp_model
    # tuned_mlp_model = classifiers.tuning_hyperparameters(model=mlp_model, model_name="MLP", x_train=X_train,
    #                                                      y_train=y_train, x_val=X_val, y_val=y_val)
    # # Evaluate MLP model
    # evaluate.evaluate_model(model=tuned_mlp_model, model_name="MLP",
    #                         x_test=X_test, y_test=y_test, test_dataset=test_dataset)
    #
    # # CNN model tuning
    # cnn_model = classifiers.build_cnn_model
    # tuned_cnn_model = classifiers.tuning_hyperparameters(model=cnn_model, model_name="CNN", x_train=X_train,
    #                                                      y_train=y_train, x_val=X_val, y_val=y_val)
    # # Evaluate CNN model
    # evaluate.evaluate_model(model=tuned_cnn_model, model_name="CNN",
    #                         x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # ***************************** #
    # **** VERSIONE PRECEDENTE **** #
    # ***************************** #

    # # Neural Network model tuning
    # nn_model = classifiers.build_nn_model
    # tuned_nn_model = classifiers.tuning_model_hyperparameter(model=nn_model, model_name="NN", x_train=X_train,
    #                                                          y_train=y_train, x_val=X_val, y_val=y_val)
    # # Evaluate NN model
    # classifiers.evaluate_model(model=tuned_nn_model, model_name="NN", x_train=X_train, y_train=y_train,
    #                            x_val=X_val, y_val=y_val, x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # MLP Model Tuning
    # mlp_model = classifiers.build_mlp_model
    # tuned_mlp_model = classifiers.tuning_model_hyperparameter(model=mlp_model, model_name="MLP", x_train=X_train,
    #                                                           y_train=y_train, x_val=X_val, y_val=y_val)
    # # Evaluate MLP model
    # classifiers.evaluate_model(model=tuned_mlp_model, model_name="MLP", x_train=X_train, y_train=y_train,
    #                            x_val=X_val, y_val=y_val, x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # CNN model tuning
    # cnn_model = classifiers.build_cnn_model
    # tuned_cnn_model = classifiers.tuning_model_hyperparameter(model=cnn_model, model_name="CNN", x_train=X_train,
    #                                                           y_train=y_train,  x_val=X_val, y_val=y_val)
    # # Evaluate CNN model
    # classifiers.evaluate_model(model=tuned_cnn_model, model_name="CNN", x_train=X_train, y_train=y_train, x_val=X_val,
    #                            y_val=y_val, x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # ResNet50 Model
    # resnet50_model = classifiers.resnet50_model(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)
    # classifiers.evaluate_model(model=resnet50_model, model_name="ResNet50", x_train=X_train, y_train=y_train,
    #                            x_val=X_val, y_val=y_val, x_test=X_test, y_test=y_test, test_dataset=test_dataset)

# My import
import classifiers
import plot_functions
import constants as const
import prepare_dataset as prepare
import utils.general_functions as general
import utils.pre_processing as pre_processing

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
    # Checking the dataset
    check_dataset = input("> START TO CHECK THE DATASET? [Y/N]: ")
    if check_dataset.upper() == "Y":
        pre_processing.checking_dataset(dataset_path=const.DATASET_PATH,
                                        train_dir_path=const.TRAIN_DIR,
                                        test_dir_path=const.TEST_DIR,
                                        show=False,
                                        save=False)

    # Load keras datasets
    train_dataset, val_dataset, test_dataset = prepare.load_dataset(train_data_dir=const.TRAIN_DIR,
                                                                    test_data_dir=const.TEST_DIR)

    # Printing information about the datasets
    print("\n> Class Names: {}".format(train_dataset.class_names) +
          "\n> Type specification: {}".format(train_dataset.element_spec))

    # Visualize the dataset with corresponding labels
    plot_functions.plot_view_dataset(train_ds=train_dataset, show_on_screen=False, store_in_folder=False)

    # Scaling data
    train_ds = prepare.data_normalization(tf_dataset=train_dataset)
    val_ds = prepare.data_normalization(tf_dataset=val_dataset)
    test_ds = prepare.data_normalization(tf_dataset=test_dataset)

    # test con array
    X_train, y_train = prepare.image_to_array(train_ds)
    X_val, y_val = prepare.image_to_array(val_ds)
    X_test, y_test = prepare.image_to_array(test_ds)

    # CNN model tuning
    cnn_model = classifiers.build_cnn_model
    tuned_cnn_model = classifiers.tuning_model_hp(model=cnn_model, model_name="CNN",
                                                  x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)
    # Evaluate CNN model
    classifiers.evaluate_model(model=tuned_cnn_model, model_name="CNN",
                               x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
                               x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # MLP Model Tuning
    # mlp_model = classifiers.build_mlp_model
    # tuned_mlp_model = classifiers.tuning_model_hp(model=mlp_model, model_name="MLP",
    #                                               x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val)
    # # Evaluate MLP model
    # classifiers.evaluate_model(model=tuned_mlp_model, model_name="MLP", x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val,
    #                            x_test=X_test, y_test=y_test, test_dataset=test_dataset)

    # # ResNet50 Model
    # resnet_model = classifiers.resnet50_model()
    # history = resnet_model.fit(train_ds, epochs=10, validation_data=val_ds)
    # plot_functions.plot_history(history=history, model_name="ResNet-50", show_on_screen=True, store_in_folder=True)
    #
    # classifiers.evaluate_model(model=resnet_model, model_name="ResNet-50", train_ds=train_ds,
    #                            val_ds=val_ds, test_ds=test_ds, test_dataset=test_dataset)


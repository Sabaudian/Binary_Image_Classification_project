# My import
import cnn_model
import constants as const
import plot_functions as my_plot
import prepare_dataset as prepare
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
    print("\n> Class Names: {}".format(train_dataset.class_names))

    # Information
    print("\n> Type specification: {}\n".format(train_dataset.element_spec))

    # Visualize the dataset with corresponding labels
    my_plot.plot_data_visualization(train_ds=train_dataset, show_on_screen=True, store_in_folder=True)

    # Scaling data
    train_ds = prepare.data_normalization(train_dataset)
    val_ds = prepare.data_normalization(val_dataset)
    test_ds = prepare.data_normalization(test_dataset)

    # CNN Model Tuning
    cnn_model.tuning_cnn(train_ds, val_ds, test_ds)


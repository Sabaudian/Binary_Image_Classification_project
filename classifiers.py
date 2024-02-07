# Import
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet import MobileNet

from sklearn.model_selection import KFold

# My import
import plot_functions
import constants as const
import prepare_dataset as prepare
import utils.general_functions as general

from models_evaluation import collect_hyperparameters_tuning_data, get_hyperparameters_search_info, evaluate_model


# *********************************************************************** #
# ************* CLASSIFIER MODELS DEFINITION AND EVALUATION ************* #
# *********************************************************************** #


# MLP model
def build_mlp_model(hp):
    """
    Build a Multi-Layer Perceptron (MLP) model with tunable hyperparameters for image binary classification.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: Keras.Model
        The compiled MLP model.
    """
    model = tf.keras.Sequential(name="MultiLayer_Perceptron")

    model.add(layers.Flatten(input_shape=const.INPUT_SHAPE, name="flatten_layer"))

    for i in range(1, 6):
        units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
        model.add(layers.Dense(units=units, activation="relu", name=f"hidden_layer_{i}"))
        # Batch Normalization for stabilization and acceleration
        model.add(layers.BatchNormalization(name=f"batch_normalization_{i}"))
        # Add dropout for regularization to prevent overfitting
        model.add(layers.Dropout(rate=0.25, name=f"dropout_{i}"))

    # Output layer with sigmoid activation for binary classification
    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "MLP"))  # Create the directory
    plot_path = os.path.join("plot", "MLP", "MLP_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    tf.keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


# CNN model
def build_cnn_model(hp):
    """
    Build a Convolutional Neural Network (CNN) model with tunable hyperparameters for image binary classification.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: tf.keras.Model
        The compiled CNN model.
    """
    # Create a Sequential model
    model = tf.keras.Sequential(name="Convolutional_Neural_Network")

    # First Convolutional layer with 32 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                            input_shape=const.INPUT_SHAPE, name="convolution_1"))
    # Batch Normalization for stabilization and acceleration
    model.add(layers.BatchNormalization(name="batch_normalization_1"))
    # MaxPooling layer to reduce spatial dimensions
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"))
    # Add dropout for regularization to prevent overfitting
    model.add(layers.Dropout(rate=0.25, name="dropout_1"))

    # Second Convolutional layer with 32 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", name="convolution_2"))
    model.add(layers.BatchNormalization(name="batch_normalization_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"))
    model.add(layers.Dropout(rate=0.25, name="dropout_2"))

    # Third Convolutional layer with 32 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="convolution_3"))
    model.add(layers.BatchNormalization(name="batch_normalization_3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(layers.Dropout(rate=0.25, name="dropout_3"))

    # Fourth Convolutional layer with 32 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="convolution_4"))
    model.add(layers.BatchNormalization(name="batch_normalization_4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_4"))
    model.add(layers.Dropout(rate=0.25, name="dropout_4"))

    # Fifth Convolutional layer with 128 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name="convolution_5"))
    model.add(layers.BatchNormalization(name="batch_normalization_5"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_5"))
    model.add(layers.Dropout(rate=0.25, name="dropout_5"))

    # Flatten layer to transition from convolutional layers to dense layer
    model.add(layers.Flatten(name="flatten"))

    # Tune the number of units in the dense layer
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation="relu", name="dense_layer"))
    # Tune the dropout rate
    hp_dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
    model.add(layers.Dropout(rate=hp_dropout_rate, name="dropout_layer"))

    # Output layer with sigmoid activation for binary classification
    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "CNN"))  # Create the directory
    plot_path = os.path.join("plot", "CNN", "CNN_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    tf.keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


# MobileNet Model
def build_mobilenet_model(hp):
    """
    Build a MobileNet model with tunable hyperparameters.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: Keras.Model
        The compiled MobileNet model.
    """
    # Load the MobileNet base model without top layers (include_top=False)
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=const.INPUT_SHAPE)

    # Freeze the weights of the MobileNet base model
    base_model.trainable = False

    # Create a Sequential model with the MobileNet base model
    model = tf.keras.Sequential(layers=[
        base_model,
        layers.Flatten(name="flatten"),
        layers.Dense(units=hp.Int("units", min_value=32, max_value=512, step=32),
                     activation="relu", name="dense_layer"),
        layers.BatchNormalization(name="batch_normalization"),
        layers.Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1), name="dropout"),
        layers.Dense(units=1, activation="sigmoid", name="output_layer")
    ], name="MobileNet")

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "MobileNet"))  # Create the directory
    plot_path = os.path.join("plot", "MobileNet", "MobileNet_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    tf.keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


# ************************************************************* #
# ****************** HYPERPARAMETER TUNING   ****************** #
# ************************************************************* #


# Perform hyperparameter Tuning
def tuning_hyperparameters(model, model_name, x_train, y_train, x_val, y_val):
    """
    Tuning the hyperparameters of the input model using Keras Tuner (kerastuner).
    The function performs hyperparameter tuning using Keras Tuner's Hyperband algorithm.
    It saves the best model, its hyperparameters, and training history in the "models" directory.

    :param model: The Keras model to be tuned.
    :param model_name: A string representing the name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    """
    # Print about the model in input
    print("\n> " + model_name + " Tuning Hyperparameters:")

    # Create a directory for the best model
    best_model_directory = os.path.join("models", model_name, "best_model")
    general.makedir(best_model_directory)

    # Best model weights filepath
    file_path = os.path.join(best_model_directory, "best_model.weights.h5")

    # Tuning model
    tuner = kt.Hyperband(
        hypermodel=model,
        objective="val_accuracy",
        max_epochs=5,
        factor=2,
        overwrite=False,
        directory="models",
        project_name=model_name
    )

    # Prints a summary of the hyperparameters in the search space
    tuner.search_space_summary()

    # Monitor "val_loss" and stop early if not improving
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    # Performs a search for best hyperparameter configurations.
    tuner.search(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val),
        callbacks=[stop_early]
    )

    # Collect hyperparameter during the tuning process
    collect_hyperparameters_tuning_data(model_name=model_name, tuner=tuner)

    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0]

    # Information about the optimal hyperparameter found during the tuning process
    get_hyperparameters_search_info(model_name=model_name, best_hyperparameters=best_hyperparameters)

    # Build the model with the optimal hyperparameters and train it on the data for 10 epochs
    print("\n> Build the model with the optimal hyperparameters and train it on the data for 10 epochs")

    optimal_hp_model = tuner.hypermodel.build(best_hyperparameters)

    history = optimal_hp_model.fit(
        x=x_train, y=y_train,
        epochs=10,
        validation_data=(x_val, y_val)
    )

    # Serialize model to JSON
    model_json = optimal_hp_model.to_json()
    with open(os.path.join(best_model_directory, "best_model.json"), "w") as json_file:
        json_file.write(model_json)

    # Save the best model's weights to the created directory
    optimal_hp_model.save_weights(filepath=file_path)

    # Check if weights have been saved
    if os.path.exists(file_path):
        print("\n> The best weights for " + model_name + " model were saved successfully!\n")

    # Plot history after tuning
    plot_functions.plot_history(history=history, model_name=model_name,
                                show_on_screen=False, store_in_folder=True)


# *************************************************************** #
# ****************** K-FOLD CROSS-VALIDATION   ****************** #
# *************************************************************** #

def zero_one_loss(y_true, y_pred):
    """
    Compute the zero-one loss metric.
    Returns 1 when y_true != y_pred, 0 otherwise.

    Parameters:
    - y_true (tensor): True labels.
    - y_pred (tensor): Predicted labels.

    Returns:
    - zero_one_loss_value (tensor): Computed zero-one loss.
    """
    # Convert predicted values to integers
    y_pred = tf.cast(y_pred + 0.5, tf.int16)
    # Convert true labels to integers
    y_true = tf.cast(y_true, tf.int16)

    # Compute the absolute difference between true and predicted labels
    diff = tf.math.abs(y_true - y_pred)

    # Check if the absolute difference is not equal to 0
    # Cast the boolean values to int16 to get 1 for mis-classification, 0 for correct classification
    zero_one_loss_value = tf.cast(tf.math.not_equal(x=diff, y=0), tf.int16)

    return zero_one_loss_value


# KFold cross validation function
def kfold_cross_validation(model_name, x_train, y_train, x_val, y_val, k_folds):
    """
    Perform K-fold cross-validation on a Keras model using the zero-one loss.

    :param model_name: String, name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    :param k_folds: Number of folds for cross-validation.
    :return model: Model after performing KFold cross-validation.
    """
    # Print about the model in input
    print("\n> " + model_name + " KFold Cross-Validation:")

    # Define the path for storing the models
    dir_path = os.path.join("models", "KFold")
    general.makedir(dir_path)
    file_path = os.path.join(dir_path, model_name + "_kfold_model.keras")

    # Check if the model.keras already exist to speed up the process
    if os.path.exists(path=file_path):

        # Load model from the right folder
        model = tf.keras.models.load_model(filepath=file_path, custom_objects={"zero_one_loss": zero_one_loss})
        print("- Model Loaded Successfully!")
    else:

        # Path to the model directory
        best_model_directory = os.path.join("models", model_name, "best_model")

        # Load json and define model
        json_file = open(os.path.join(best_model_directory, "best_model.json"), "r")
        load_model_json = json_file.read()
        json_file.close()

        # This is the structure of the best model obtained after tuning
        model = tf.keras.models.model_from_json(load_model_json)

        # Load weight into the model, i.e., the best hyperparameters
        model.load_weights(filepath=os.path.join(best_model_directory, "best_model.weights.h5"))
        print("-- Best weights for " + model_name + " model have been Loaded Successfully!")

        # KFold Cross-Validation function
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # Combine training and validation data for K-fold cross-validation
        X = np.concatenate((x_train, x_val), axis=0)
        Y = np.concatenate((y_train, y_val), axis=0)

        # Initialize lists to save data
        fold_data = []
        fold_history = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print("\n> Fold {}/{}".format(fold + 1, k_folds))

            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Create and compile a new instance of the model
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(),
                loss="binary_crossentropy",
                metrics=["accuracy", zero_one_loss]
            )

            # Train the model on the training set for this fold
            history = model.fit(
                X_train, Y_train,
                batch_size=const.BATCH_SIZE,
                epochs=10,
                validation_data=(X_val, Y_val),
                verbose=1
            )

            # Collect training history data for generate a plot later
            fold_history.append(history)

            # Brief evaluation per epoch on validation set
            val_loss, val_accuracy, val_zero_one_loss = model.evaluate(X_val, Y_val, verbose=0)

            # Collect data per fold
            fold_data.append({
                "Fold": fold + 1,
                "Loss": val_loss,
                "Accuracy (%)": val_accuracy * 100,
                "0-1 Loss": val_zero_one_loss
            })

        # Create a pandas DataFrame from the collected data
        fold_df = pd.DataFrame(data=fold_data)

        # Calculate average values
        avg_values = {
            "Fold": "Average",
            "Loss": np.mean(fold_df["Loss"]),
            "Accuracy (%)": np.mean(fold_df["Accuracy (%)"]),
            "0-1 Loss": np.mean(fold_df["0-1 Loss"])
        }

        # Concatenate the average values to the DataFrame
        fold_df = pd.concat(objs=[fold_df, pd.DataFrame([avg_values])], ignore_index=True)

        # Save fold data to csv file
        fold_data_csv_file_path = os.path.join(const.DATA_PATH, f"{model_name}_fold_data.csv")
        fold_df.to_csv(fold_data_csv_file_path, index=False, float_format="%.3f")

        # Plot fold history
        plot_functions.plot_fold_history(fold_history=fold_history, model_name=model_name,
                                         show_on_screen=False, store_in_folder=True)

        # Save the model to the path
        model.save(file_path)

    return model


# ******************************************************** #
# ****************** PROCESS WORKFLOW   ****************** #
# ******************************************************** #


# Organize the various procedures
def classification_procedure_workflow(models, x_train, y_train, x_val, y_val, x_test, y_test, kfold,
                                      random_prediction, show_plot, save_plot):
    """
    Tune hyperparameters for a dictionary of classification models,
    apply KFold cross-validation and then evaluate the various models

    :param models: A dictionary containing classification models.
    :param x_train: Training data features.
    :param y_train: Training data labels.
    :param x_val: Validation data features.
    :param y_val: Validation data labels.
    :param x_test: Test data features.
    :param y_test: Test data labels.
    :param kfold: Number of folds for K-Fold Cross-Validation.
        Default is 5.
    :param random_prediction: If True, pick random images for the prediction visualization plot.
        Default is False.
    :param show_plot: If True, displays the plot on the screen.
        Default is True.
    :param save_plot: If True, save the plot.
        Default is True.
    """
    # List to collect models data
    all_models_data = []

    # Scroll through the dictionary
    for key, value in models.items():
        # MLP, CNN and MobileNet name string
        model_name = key
        # Models
        model_type = value

        # Best model folder path
        tuned_model_folder = os.path.join("models", model_name, "best_model")
        # Best model filepath
        tuned_file_path = os.path.join(tuned_model_folder, "best_model.weights.h5")

        # Redo Tuning if not already done it
        if not os.path.exists(tuned_file_path):
            # Tuning Hyperparameters and Save the Best
            tuning_hyperparameters(model=model_type, model_name=model_name,
                                   x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

        # Apply Kfold Cross-validation
        kfold_result = kfold_cross_validation(model_name=model_name,
                                              x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, k_folds=kfold)

        # Evaluate the results on the Test set
        data = evaluate_model(model=kfold_result, model_name=model_name, x_test=x_test, y_test=y_test,
                              random_prediction=random_prediction, show_plot=show_plot, save_plot=save_plot)
        all_models_data.append(data)

    # Create a pandas DataFrame
    df = pd.DataFrame(all_models_data)

    # Save the data
    general.makedir(dirpath=const.DATA_PATH)
    file_path = os.path.join(const.DATA_PATH, "Models_Performances_on_Test_Set.csv")
    df.to_csv(file_path, index=False, float_format="%.3f")


# To be called in the main
def classification_and_evaluation(train_path, test_path, random_prediction=False, show_plot=True, save_plot=True):
    """
    Perform classification and evaluation of image datasets.

    This function loads image datasets, prints information about class names, visualizes the dataset, scales the data,
    performs data augmentation, converts images to arrays,
    retrieves classification models, and applies KFold tuning.
    The classification models are evaluated on the training, validation,
    and testing datasets, and the results are displayed
    through plots if specified.

    :param train_path: path to train data set.
    :param test_path: path to test data set.
    :param random_prediction: If True, pick random images for the prediction visualization plot.
        Default is False.
    :param show_plot: If True, displays the plot on the screen.
        Default is True.
    :param save_plot: If True, save the plot.
        Default is True.
    """
    # Load keras datasets
    train_dataset, val_dataset, test_dataset = prepare.load_dataset(train_data_dir=train_path,
                                                                    test_data_dir=test_path)

    # Printing information about the datasets
    print("\n> Class Names:"
          "\n\t- Class 0 = {}"
          "\n\t- Class 1 = {}".format(train_dataset.class_names[0], train_dataset.class_names[1]))

    # Visualize the dataset showing some images with corresponding labels
    plot_functions.plot_view_dataset(train_ds=train_dataset, show_on_screen=show_plot, store_in_folder=save_plot)

    # Scaling data
    train_ds = prepare.data_normalization(tf_dataset=train_dataset, augment=True)
    val_ds = prepare.data_normalization(tf_dataset=val_dataset, augment=False)
    test_ds = prepare.data_normalization(tf_dataset=test_dataset, augment=False)

    # Visualize the data_augmentation process effect
    plot_functions.plot_data_augmentation(train_ds=train_dataset, data_augmentation=prepare.perform_data_augmentation(),
                                          show_on_screen=show_plot, store_in_folder=save_plot)

    # dataset into array
    X_train, y_train = prepare.image_to_array(train_ds)
    X_val, y_val = prepare.image_to_array(val_ds)
    X_test, y_test = prepare.image_to_array(test_ds)

    # Get the classification models
    classification_models = general.get_classifier()

    # Tuning, apply K-Fold Cross-Validation and then evaluate the models
    classification_procedure_workflow(models=classification_models, x_train=X_train, y_train=y_train, x_val=X_val,
                                      y_val=y_val, x_test=X_test, y_test=y_test, kfold=const.KFOLD,
                                      random_prediction=random_prediction, show_plot=show_plot, save_plot=save_plot)

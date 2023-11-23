# Import
import os
import pathlib
import time
import keras
import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from IPython import display
from keras import layers
from keras import backend
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_squared_error

# My import
import plot_functions
import constants as const
import utils.general_functions as general


# ******************************************************** #
# ************* CLASSIFIER MODELS DEFINITION ************* #
# ******************************************************** #


# NN Model
def build_nn_model(hp):
    """
    Build a simple Neural Network (NN) model with tunable hyperparameters for image binary classification.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: Keras.Model
        The compiled MLP model.
    """

    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=const.INPUT_SHAPE))

    # Tune the number of units in the dense layer
    # Choose an optimal value between 32-512
    for i in range(1, 3):
        units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
        model.add(layers.Dense(units=units, activation="relu", name=f"hidden_layer_{i}"))

    # Output layer
    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "NN"))  # Create the directory
    plot_path = os.path.join("plot", "NN", "NN_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


# MLP model
def build_mlp_model(hp):
    """
    Build a Multi-Layer Perceptron (MLP) model with tunable hyperparameters for image binary classification.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: Keras.Model
        The compiled MLP model.
    """
    model = tf.keras.Sequential()

    model.add(layers.Flatten(input_shape=const.INPUT_SHAPE))

    for i in range(1, 6):
        units = hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32)
        model.add(layers.Dense(units=units, activation="relu", name=f"hidden_layer_{i}"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

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
    keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

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
    model = tf.keras.Sequential()

    # Convolutional layer with 32 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                            input_shape=const.INPUT_SHAPE, name="convolution_1"))
    # Batch Normalization for stabilization and acceleration
    model.add(layers.BatchNormalization())
    # MaxPooling layer to reduce spatial dimensions
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # Convolutional layer with 64 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="convolution_2"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # Convolutional layer with 128 filters and a kernel size of (3, 3)
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name="convolution_3"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"))

    # Flatten layer to transition from convolutional layers to dense layer
    model.add(layers.Flatten(name="flatten"))

    # Tune the number of units in the dense layer
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation="relu", name="hidden_layer"))

    # Add dropout for regularization to prevent overfitting
    model.add(layers.Dropout(0.5))

    # Output layer with sigmoid activation for binary classification
    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "CNN"))  # Create the directory
    plot_path = os.path.join("plot", "CNN", "CNN_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


# VGG-16 Model
def build_vgg16_model(hp):
    """
    Build a VGG16 model with tunable hyperparameters for image binary classification.

    :param hp: Keras.utils.HyperParameters.
        Hyperparameters for model tuning.

    :return: Keras.Model
        The compiled VGG16 model.
    """
    # Load the VGG16 base model without top layers (include_top=False)
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=const.INPUT_SHAPE)

    # Freeze the weights of the VGG16 base model
    base_model.trainable = False

    # Create a Sequential model with the VGG16 base model
    model = tf.keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(units=hp.Int("units", min_value=128, max_value=1024, step=128), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)),
        layers.Dense(1, activation="sigmoid")
    ])

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "VGG16"))  # Create the directory
    plot_path = os.path.join("plot", "VGG16", "VGG16_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

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
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(units=hp.Int("units", min_value=128, max_value=1024, step=128), activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)),
        layers.Dense(1, activation="sigmoid")
    ])

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Store and Display the model's architecture
    general.makedir(os.path.join("plot", "MobileNet"))  # Create the directory
    plot_path = os.path.join("plot", "MobileNet", "MobileNet_model_summary_plot.jpg")  # Path to store the plot
    model.summary()
    keras.utils.plot_model(model=model, to_file=plot_path, dpi=96)

    return model


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

    # Create a directory for the best model
    best_model_directory = os.path.join("models", model_name, "best_model")
    general.makedir(best_model_directory)

    # Best model filepath
    file_path = os.path.join(best_model_directory, "best_model.h5")

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

    # Print search summary
    tuner.search_space_summary()

    # Monitor "val_loss" and stop early if not improving
    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

    # Performs a search for best hyperparameter configurations.
    tuner.search(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val),
        callbacks=[stop_early]
    )
    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    if model_name == "NN":
        print("\n> The hyperparameter search is complete!")
        for i in range(1, 2):
            print("- The optimal number of units in hidden_layer_{} is: {}"
                  .format(i, best_hyperparameters[f"units_{i}"]))

        print("- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))
    elif model_name == "MLP":
        print("\n> The hyperparameter search is complete!")
        for i in range(1, 6):
            print("\t- The optimal number of units in hidden_layer_{} is: {}"
                  .format(i, best_hyperparameters[f"units_{i}"]))

        print("- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))
    elif model_name == "CNN":
        print("\n> The hyperparameter search is complete!"
              "\n- The optimal number of units in the densely-connected layer is: {}"
              .format(best_hyperparameters.get("units")) +
              "\n- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))
    else:
        print("\n> The hyperparameter search is complete!"
              "\n- The optimal number of units in the densely-connected layer is: {}"
              .format(best_hyperparameters.get("units")) +
              "\n- The optimal dropout rate is: {}"
              .format(best_hyperparameters.get("dropout_rate")) +
              "\n- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))

    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 10 epochs
    print("\n> Build the model with the optimal hyperparameters and train it on the data for 10 epochs")

    optimal_hp_model = tuner.hypermodel.build(best_hyperparameters)

    history = optimal_hp_model.fit(
        x=x_train, y=y_train,
        epochs=10,
        validation_data=(x_val, y_val)
    )

    # Compute best epoch value
    val_accuracy_per_epochs = history.history["val_accuracy"]
    best_epoch = val_accuracy_per_epochs.index(max(val_accuracy_per_epochs)) + 1
    print("\n> Best Epoch: {}\n".format(best_epoch, ))

    # Rerun the model with the optimal value for the epoch
    hypermodel = tuner.hypermodel.build(best_hyperparameters)
    # Retrain the model
    hypermodel_history = hypermodel.fit(
        x=x_train, y=y_train,
        epochs=best_epoch,
        validation_data=(x_val, y_val)
    )

    # Serialize model to JSON
    model_json = hypermodel.to_json()
    with open(os.path.join(best_model_directory, "best_model.json"), "w") as json_file:
        json_file.write(model_json)

    # Save the best model's weights to the created directory
    hypermodel.save_weights(filepath=file_path)

    # Plot history after tuning
    plot_functions.plot_history(history=hypermodel_history, model_name=model_name,
                                show_on_screen=True, store_in_folder=True)


# # zero one loss
# def zero_one_loss_funtion(true_label, pred_label):
#     # Transforms predicted probabilities into binary labels (0 o 1)
#     pred_binary = tf.round(pred_label)
#
#     # Compare the predicted binary labels with the actual labels
#     errors = tf.math.abs(true_label - pred_binary)
#
#     # Calculate the zero-one loss as the average of the errors
#     zero_one_loss = tf.reduce_mean(errors)
#
#     return zero_one_loss


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

    # Initialize lists to save data
    fold_history = []
    fold_loss = []
    fold_accuracy = []
    fold_zero_one_loss = []

    # Define the path for storing the models
    dir_path = os.path.join("models", "KFold")
    general.makedir(dir_path)
    file_path = os.path.join(dir_path, model_name + "_kfold_model.h5")

    try:
        model = keras.models.load_model(file_path)
        print("-- Model Loaded Successfully!")
    except (OSError, IOError):

        # Path to the model directory
        best_model_directory = os.path.join("models", model_name, "best_model")

        # Load json and define model
        json_file = open(os.path.join(best_model_directory, "model.json"), "r")
        load_model_json = json_file.read()
        json_file.close()

        # This is the structure of the best model obtained after tuning
        model = keras.models.model_from_json(load_model_json)

        # Load weight into the model, i.e., the best hyperparameters
        model.load_weights(filepath=os.path.join(best_model_directory, "best_model.h5"))
        print("-- Best Hyperparameters for " + model_name + " model have been Loaded Successfully!")

        # KFold Cross-Validation function
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Combine training and validation data for K-fold cross-validation
        x_combined = np.concatenate((x_train, x_val), axis=0)
        y_combined = np.concatenate((y_train, y_val), axis=0)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_combined, y_combined)):
            print("\n> Fold {}/{}".format(fold + 1, k_folds))

            X_train, X_val = x_combined[train_idx], x_combined[val_idx]
            Y_train, Y_val = y_combined[train_idx], y_combined[val_idx]

            # Create and compile a new instance of the model
            model.compile(
                optimizer=keras.optimizers.legacy.Adam(),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Monitor "val_loss" and stop early if not improving
            stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

            # Train the model on the training set for this fold
            history = model.fit(
                X_train, Y_train,
                epochs=10,
                validation_data=(X_val, Y_val),
                callbacks=stop_early,
                verbose=1
            )

            # # Plot history after tuning
            # plot_functions.plot_history(history=history, model_name=model_name + f" Fold {fold + 1}",
            #                             show_on_screen=False, store_in_folder=False)

            # Evaluate the model on the validation set for this fold
            y_predicts = model.predict(X_val)
            predictions = (y_predicts > 0.5).astype("int")

            val_zero_one_loss = zero_one_loss(Y_val, predictions)
            val_loss, val_accuracy = model.evaluate(X_val, Y_val)

            # Print and store the results for this fold
            print("- Loss: {}\n"
                  "- Accuracy: {}\n"
                  "- Zero-one Loss: {}".format(val_loss, val_accuracy, val_zero_one_loss))
            print("__________________________________________________________________________________________")

            fold_loss.append(val_loss)
            fold_accuracy.append(val_accuracy)
            fold_history.append(history)
            fold_zero_one_loss.append(val_zero_one_loss)

        # Save the model to the path
        model.save(file_path)

    # # Calculate and print the mean and standard deviation of the evaluation metrics across folds
    # mean_zero_one_loss = np.mean(fold_zero_one_loss)
    # mean_loss = np.mean(fold_loss)
    # mean_accuracy = np.mean(fold_accuracy)
    # standard_deviation_zero_one_loss = np.std(fold_zero_one_loss)
    # standard_deviation_loss = np.std(fold_loss)
    # standard_deviation_accuracy = np.std(fold_accuracy)
    #
    # print("\n> " + model_name + " KFold data:")
    # print("- Mean Zero-One Loss: {}".format(mean_zero_one_loss))
    # print("- Mean Loss: {}".format(mean_loss))
    # print("- Mean Accuracy: {}".format(mean_accuracy))
    # print("- Standard Deviation Zero-One Loss: {}".format(standard_deviation_zero_one_loss))
    # print("- Standard Deviation Loss: {}".format(standard_deviation_loss))
    # print("- Standard Deviation Accuracy: {}".format(standard_deviation_accuracy))

    return model


# ******************************************** #
# ************* MODEL EVALUATION ************* #
# ******************************************** #


# def accuracy_loss_model_dict(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test):
#     """
#     Compute a simple evaluation of the model,
#     printing and saving the loss and accuracy for the train, val and test set.
#     The data are saved as a CSV file.
#     :param model: Model in input.
#     :param model_name: Name of the model in input.
#     :param x_train: Input values of the train dataset.
#     :param y_train: Target values of the train dataset.
#     :param x_val: Input values of the val dataset.
#     :param y_val: Target values of the val dataset.
#     :param x_test: Input values of the test dataset.
#     :param y_test: Target values of the test dataset.
#     """
#     # Compute loss and accuracy
#     train_loss, train_accuracy = model.evaluate(x=x_train, y=y_train, verbose=0)
#     val_loss, val_accuracy = model.evaluate(x=x_val, y=y_val, verbose=0)
#     test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
#
#     # Print evaluation info. about the model
#     print("\t- Train Loss: {:.3f}%"
#           "\n\t- Train Accuracy: {:.3f}%"
#           "\n\t- Val Loss: {:.3f}%"
#           "\n\t- Val Accuracy: {:.3f}%"
#           "\n\t- Test Loss: {:.3f}%"
#           "\n\t- Test Accuracy: {:.3f}%\n"
#           .format(train_loss * 100, train_accuracy * 100, val_loss * 100,
#                   val_accuracy * 100, test_loss * 100, test_accuracy * 100))
#
#     # declare dictionary for saving evaluation data about the model in input
#     dictionary = {
#         "Train_Loss": [],
#         "Train_Accuracy": [],
#         "Val_Loss": [],
#         "Val_Accuracy": [],
#         "Test_Loss": [],
#         "Test_Accuracy": []
#     }
#
#     # Save information into a csv file
#     dictionary.update({
#         "Train_Loss": [train_loss],
#         "Train_Accuracy": [train_accuracy],
#         "Val_Loss": [val_loss],
#         "Val_Accuracy": [val_accuracy],
#         "Test_Loss": [test_loss],
#         "Test_Accuracy": [test_accuracy],
#     })
#
#     # # Save dictionary as file .csv
#     # df = pd.DataFrame.from_dict(dictionary)
#     # df.to_csv("data/" + model_name + "_model_simple_loss_accuracy_eval.csv", index=False, float_format="%.3f")

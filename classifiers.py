# Import
import os
import time
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from IPython import display
from keras import layers
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
    Tuning the hyperparameters of the model in input.
    :param model: Model in input.
    :param model_name: String, name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    :return: Trained model with tuned hyperparameter.
    """

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

    # Create a directory for the best model within the project directory
    best_model_directory = os.path.join("models", model_name, "best_model")
    # os.makedirs(best_model_directory, exist_ok=True)
    general.makedir(best_model_directory)
    # Save the best model's weights to the created directory
    hypermodel.save_weights(filepath=os.path.join(best_model_directory, "best_model"))

    # Plot history after tuning
    plot_functions.plot_history(history=hypermodel_history, model_name=model_name,
                                show_on_screen=True, store_in_folder=True)
    return hypermodel


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
def kfold_cross_validation(model, model_name, x_train, y_train, x_val, y_val, k_folds):
    """
    Perform K-fold cross-validation on a Keras model using the zero-one loss.

    :param model: Model in input.
    :param model_name: String, name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    :param k_folds: Number of folds for cross-validation.
    :return model: Model after performing KFold cross-validation.
    """

    best_model_directory = os.path.join("models", model_name, "best_model")
    model.load_weights(filepath=os.path.join(best_model_directory, "best_model"))

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_zero_one_loss = []
    fold_accuracy = []

    # Combine training and validation data for K-fold cross-validation
    x_combined = np.concatenate((x_train, x_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_combined, y_combined)):
        print(f"\nFold {fold + 1}/{k_folds}")

        # Create and compile a new instance of the model
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Train the model on the training set for this fold
        kfold_history = model.fit(
            x_combined[train_idx], y_combined[train_idx],
            epochs=10,
            validation_data=(x_combined[val_idx], y_combined[val_idx]),
            verbose=1
        )

        # Plot history after tuning
        plot_functions.plot_history(history=kfold_history, model_name=model_name + f" fold {fold + 1}",
                                    show_on_screen=True, store_in_folder=False)

        # Evaluate the model on the validation set for this fold
        predictions = model.predict(x_combined[val_idx])
        zero_one_loss_value = zero_one_loss(y_combined[val_idx], np.argmax(predictions, axis=1))

        accuracy = model.evaluate(x_combined[val_idx], y_combined[val_idx])[1]

        # Print and store the results for this fold
        print(f"Zero-One Loss: {zero_one_loss_value}, Validation Accuracy: {accuracy}")
        fold_zero_one_loss.append(zero_one_loss_value)
        fold_accuracy.append(accuracy)

    # Calculate and print the mean and standard deviation of the evaluation metrics across folds
    mean_zero_one_loss = np.mean(fold_zero_one_loss)
    mean_accuracy = np.mean(fold_accuracy)

    print("\nMean Zero-One Loss:", mean_zero_one_loss)
    print("Mean Accuracy:", mean_accuracy)
    print("Standard Deviation Zero-One Loss:", np.std(fold_zero_one_loss))
    print("Standard Deviation Accuracy:", np.std(fold_accuracy))

    return model

# FUNZIONE DA RICHIAMARE NEL MAIN
# def classification_and_evaluation():


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
#
#
# def compute_evaluation_metrics(model, model_name, x_test, y_test):
#     """
#     Get the classification report about the model in input.
#     :param model: Model in input.
#     :param model_name: Name of the model.
#     :param x_test: Input values of the test dataset.
#     :param y_test: Target values of the test dataset.
#     :return: The Classification Report Dataframe of the model
#     """
#     # Predict the target vector
#     predict = model.predict(x_test)
#
#     # Convert the predictions to binary classes (0 or 1)
#     predictions = (predict >= 0.5).astype(int)
#     predictions = predictions.flatten()
#
#     # Compute report
#     clf_report = classification_report(y_test, predictions, target_names=const.CLASS_LIST, digits=2, output_dict=True)
#
#     # Update so in df is shown in the same way as standard print
#     clf_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": clf_report["accuracy"],
#                                     "support": clf_report.get("macro avg")["support"]}})
#     df = pd.DataFrame(clf_report).transpose()
#
#     # Print report
#     print("\n> Classification Report:")
#     display.display(df)
#     print("\n")
#
#     # Save the report
#     general.makedir(const.DATA_PATH + "/" + const.REPORT_PATH)
#     df.to_csv(const.DATA_PATH + "/" + const.REPORT_PATH + "/" + model_name + "_classification_report.csv",
#               index=True, float_format="%.2f")
#     return df
#
#
# # Model evaluation with extrapolation of data and information plot
# def evaluate_model(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test, test_dataset):
#     """
#     Evaluate the Performance of the model on the test set.
#     :param model: The model in input.
#     :param model_name: Name of the model.
#     :param x_train: Input values of the training dataset.
#     :param y_train: Target values of the training dataset.
#     :param x_val: Input values of the validation dataset.
#     :param y_val: Target values of the validation dataset.
#     :param x_test: Input values of the test dataset.
#     :param y_test: Target values of the test dataset.
#     :param test_dataset: Raw keras dataset (tf.keras.utils.image_dataset_from_directory).
#     """
#
#     # Evaluate the model
#     print("\n> " + model_name + " Model Evaluation:")
#
#     # Compute a simple evaluation report on the model performances
#     accuracy_loss_model_dict(model=model, model_name=model_name,
#                              x_train=x_train, y_train=y_train,
#                              x_val=x_val, y_val=y_val,
#                              x_test=x_test, y_test=y_test)
#
#     # Compute the classification_report of the model
#     compute_evaluation_metrics(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
#
#     # Plot Confusion Matrix
#     plot_functions.plot_confusion_matrix(model=model, model_name=model_name, x_test=x_test, y_test=y_test,
#                                          show_on_screen=True, store_in_folder=True)
#
#     # Plot a representation of the prediction
#     plot_functions.plot_predictions_evaluation(model=model, model_name=model_name,
#                                                class_list=const.CLASS_LIST, x_test=x_test, y_test=y_test,
#                                                show_on_screen=True, store_in_folder=True)
#
#     # Plot a visual representation of the classification model, predicting classes
#     plot_functions.plot_visual_prediction(model=model, model_name=model_name, x_test=x_test, test_dataset=test_dataset,
#                                           show_on_screen=True, store_in_folder=True)

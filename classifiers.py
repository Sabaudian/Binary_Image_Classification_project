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
    Building a simple Neural Network model
    :param hp: parameter used for hyperparameter tuning
    :return: The compiled model
    """
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=const.INPUT_SHAPE))

    # Tune the number of units in the dense layer
    # Choose an optimal value between 32-512
    hp_units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units_1, activation="relu", name="hidden_layer_1"))

    # Tune the number of units in the dense layer
    # Choose an optimal value between 32-512
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units_2, activation="relu", name="hidden_layer_2"))

    # Output layer
    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    # Compile
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Print and Save the model summary
    model.summary()
    keras.utils.plot_model(model=model, to_file="plot/NN_model_summary_plot.png", dpi=300)

    return model


# MLP model
def build_mlp_model(hp):
    """
    Building a Multi-Layer Perceptron model
    :param hp: Parameter used for hyperparameter tuning
    :return: The compiled model
    """
    model = tf.keras.Sequential()

    model.add(layers.Flatten(input_shape=const.INPUT_SHAPE))
    model.add(layers.Dense(units=256, activation="relu", name="hidden_layer_1"))
    model.add(layers.Dense(units=128, activation="relu", name="hidden_layer_2"))
    model.add(layers.Dense(units=64, activation="relu", name="hidden_layer_3"))
    model.add(layers.Dense(units=32, activation="relu", name="hidden_layer_4"))

    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units, activation="relu", name="hidden_layer_5"))

    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Print and Save the model summary
    model.summary()
    keras.utils.plot_model(model=model, to_file="plot/MLP_model_summary_plot.png", dpi=300)

    return model


# CNN model
def build_cnn_model(hp):
    """
    Building a Convolutional Neural Network model
    :param hp: parameter used for hyperparameter tuning
    :return: The compiled model
    """
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                            input_shape=const.INPUT_SHAPE, name="convolution_1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="convolution_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name="convolution_3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(layers.Flatten(name="flatten"))

    # Tune the number of units in the dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)

    model.add(layers.Dense(units=hp_units, activation="relu", name="hidden_layer"))

    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    # Compile
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Print and Save the model summary
    model.summary()
    keras.utils.plot_model(model=model, to_file="plot/CNN_model_summary_plot.png", dpi=300)

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
    # Stop training when a monitored metric (i.e.: val_loss) has stopped improving.
    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

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
        print("\n> The hyperparameter search is complete!"
              "\n\t- The optimal number of units in the First densely-connected layer is: {}"
              .format(best_hyperparameters.get("units_1")) +
              "\n\t- The optimal number of units in the Second densely-connected layer is: {}"
              .format(best_hyperparameters.get("units_2")) +
              "\n\t- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))
    else:
        print("\n> The hyperparameter search is complete!"
              "\n\t- The optimal number of units in the densely-connected layer is: {}"
              .format(best_hyperparameters.get("units")) +
              "\n\t- The optimal learning rate for the optimizer is: {}"
              .format(best_hyperparameters.get("learning_rate")))

    # Build the model with the optimal hyperparameters and train it on the data for 10 epochs
    print("\n> Build the model with the optimal hyperparameters and train it on the data for 10 epochs\n")
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
    best_model_directory = "models/" + model_name + "/best_model"
    os.makedirs(best_model_directory, exist_ok=True)

    # Save the best model's weights to the created directory
    hypermodel.save_weights(os.path.join(best_model_directory, "best_model"))

    # Plot history after tuning
    plot_functions.plot_history(history=hypermodel_history, model_name=model_name,
                                show_on_screen=True, store_in_folder=True)
    return hypermodel


# # ResNet50 model
# def resnet50_model(x_train, y_train, x_val, y_val):
#     """
#     Configure a ResNet50 model
#     :return: compiled model
#     """
#     # ResNet50 model
#     model = ResNet50(weights="imagenet", include_top=False, input_shape=const.INPUT_SHAPE)
#
#     # Model summary
#     model.summary()
#     keras.utils.plot_model(model=model, to_file="plot/ResNet50_model.png", dpi=300)
#
#     for layer in model.layers:
#         layer.trainable = False
#     x = model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(units=128, activation="relu")(x)
#     predict = layers.Dense(units=1, activation="sigmoid")(x)
#
#     resnet50 = keras.Model(inputs=model.input, outputs=predict)
#
#     resnet50.compile(
#         optimizer="adam",
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )
#
#     history = resnet50.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_val, y_val))
#     plot_functions.plot_history(history=history, model_name="ResNet50", show_on_screen=True, store_in_folder=False)
#
#     return resnet50

# def zero_one_loss_funtion(y_true, y_pred):
#     # Transforms predicted probabilities into binary labels (0 o 1)
#     y_pred_binary = tf.round(y_pred)
#
#     # Compare the predicted binary labels with the actual labels
#     errors = tf.math.abs(y_true - y_pred_binary)
#
#     # Calculate the zero-one loss as the average of the errors
#     zero_one_loss = tf.reduce_mean(errors)
#
#     return zero_one_loss
#
#
# def kfold_cross_validation(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test):
#
#
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         loss="binary_crossentropy",
#         metrics=[zero_one_loss_funtion]
#     )
#


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

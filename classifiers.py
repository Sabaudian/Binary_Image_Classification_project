# Import
import time
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from keras import layers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_squared_error

# My import
import plot_functions
import constants as const
import utils.general_functions


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

    # Tune the number of units in the first dense layer
    # Choose an optimal value between 32-512
    hp_units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units_1, activation="relu"))

    # Tune the number of units in the first dense layer
    # Choose an optimal value between 32-512
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
    model.add(layers.Dense(units=hp_units_2, activation="relu"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    # Plot model summary
    # Save plot into chosen directory
    # model.summary()
    # keras.utils.plot_model(model=model, to_file="plot/NN_model_summary.png", dpi=300)

    # Compile
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def tuning_nn_model(model, model_name, train_ds, val_ds, test_ds):
    tuner = kt.RandomSearch(
        hypermodel=model,
        objective="val_accuracy",
        max_trials=5,
        executions_per_trial=3,
        directory="models",
        project_name=model_name)

    # Running the tuner
    tuner.search(train_ds, epochs=10, validation_data=val_ds)

    # Retrieving the best model
    best_models = tuner.get_best_models(num_models=1)

    test_loss, test_acc = best_models[0].evaluate(test_ds)
    print("Test accuracy:", test_acc)


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

    # Print and save model summary
    model.summary()
    keras.utils.plot_model(model=model, to_file="plot/CNN_model_summary.png", dpi=300)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    # Compile
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
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

    model.summary()
    keras.utils.plot_model(model=model, to_file="plot/plot_MLP_model_summary.png", dpi=300)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Perform hyperparameter Tuning
def tuning_model_hp(model, model_name, x_train, y_train, x_val, y_val):
    """
    Tuning the hyperparameter of the CNN model.
    :param model: Compiled model in input.
    :param model_name: String, name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    :return: Trained model with tuned hyperparameter.
    """
    # Tuning CNN
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
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[stop_early])
    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\n> The hyperparameter search is complete!"
          "\n- The optimal number of units in the first densely-connected layer is: {}"
          .format(best_hps.get("units")) +
          "\n- The optimal learning rate for the optimizer is: {}\n".format(best_hps.get("learning_rate")))

    # Build the model with the optimal hyperparameters and train it on the data for 10 epochs
    print("\n> Build the model with the optimal hyperparameters and train it on the data for 10 epochs\n")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_val, y_val))

    # Compute best epoch value
    val_accuracy_per_epochs = history.history["val_accuracy"]
    best_epoch = val_accuracy_per_epochs.index(max(val_accuracy_per_epochs)) + 1
    print("\n> Best Epoch: {}\n".format(best_epoch, ))

    # Rerun the model with the optimal value for the epoch
    my_hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel_history = my_hypermodel.fit(x=x_train, y=y_train, epochs=best_epoch, validation_data=(x_val, y_val))

    # Plot history after tuning
    plot_functions.plot_history(history=hypermodel_history, model_name=model_name,
                                show_on_screen=False, store_in_folder=False)
    return my_hypermodel


# # ResNet50 model
# def resnet50_model():
#     """
#     Configure a Multi-Layer Perceptron model
#     :return: resnet50 model
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
#     return resnet50


def prediction_comparison(model, X_test, y_test):
    # Predict the target vector
    y_predict = model.predict(X_test)
    # Genres
    genres = {i: const.CLASS_LIST[i] for i in range(0, len(const.CLASS_LIST))}

    clf_data = pd.DataFrame(columns=["real_genre_num", "predict_genre_num",
                                     "real_genre_label", "predict_genre_label"])
    clf_data["real_genre_num"] = y_test.astype(int)
    clf_data["predict_genre_num"] = y_predict.astype(int)

    # compare real values with predicted values
    comparison_column = np.where(clf_data["real_genre_num"] == clf_data["predict_genre_num"], True, False)
    clf_data["check"] = comparison_column

    clf_data["real_genre_label"] = clf_data["real_genre_num"].replace(genres)
    clf_data["predict_genre_label"] = clf_data["predict_genre_num"].replace(genres)

    input_data = pd.DataFrame()
    input_data[["Genre", "Real_Value"]] = \
        clf_data[["real_genre_label", "predict_genre_label"]].groupby(["real_genre_label"], as_index=False).count()
    input_data[["Genre", "Predict_Value"]] = \
        clf_data[["real_genre_label", "predict_genre_label"]].groupby(["predict_genre_label"], as_index=False).count()

    return input_data


def evaluate_model(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test, test_dataset):
    """
    Evaluate the Performance of the model on the test set.
    :param model: The model in input.
    :param model_name: Name of the model.
    :param x_train: Input values of the training dataset.
    :param y_train: Target values of the training dataset.
    :param x_val: Input values of the validation dataset.
    :param y_val: Target values of the validation dataset.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    :param test_dataset: Raw keras dataset (tf.keras.utils.image_dataset_from_directory).
    """
    # declare dictionary for saving evaluation data about the model in input
    model_evaluation_dict = {
        "Train_Loss": [],
        "Train_Accuracy": [],
        "Val_Loss": [],
        "Val_Accuracy": [],
        "Test_Loss": [],
        "Test_Accuracy": []
    }

    # Evaluate the model
    print("\n> Model Evaluation:")

    train_loss, train_accuracy = model.evaluate(x=x_train, y=y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(x=x_val, y=y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)

    print("> Train Loss: {}"
          "\n> Train Accuracy: {}\n"
          "\n> Val Loss: {}"
          "\n> Val Accuracy: {}\n"
          "\n> Test Loss: {}"
          "\n> Test Accuracy: {}\n"
          .format(train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy))

    # Save information into a csv file
    model_evaluation_dict.update({
        "Train_Loss": [train_loss],
        "Train_Accuracy": [train_accuracy],
        "Val_Loss": [val_loss],
        "Val_Accuracy": [val_accuracy],
        "Test_Loss": [test_loss],
        "Test_Accuracy": [test_accuracy],
    })
    df = pd.DataFrame.from_dict(model_evaluation_dict)
    df.to_csv("data/" + model_name + "_model_evaluation.csv", index=False, float_format="%.3f")

    # TEST clf report
    pred_prob = model.predict(x_test)
    predictions = (pred_prob > 0.5).astype(int)
    predictions = predictions.flatten()
    print("\n> Classification Report:\n\n{}".format(
        classification_report(y_test, predictions, target_names=test_dataset.class_names)))

    # Plot Confusion Matrix
    plot_functions.plot_confusion_matrix(model=model, model_name=model_name, x_test=x_test, y_test=y_test,
                                         show_on_screen=True, store_in_folder=False)

    # pred = prediction_comparison(model=model, X_test=x_test, y_test=y_test)
    # plot_functions.plot_predictions_evaluation(input_data=pred, model_name=model_name, class_list=const.CLASS_LIST,
    #                                            show_on_screen=True, store_in_folder=False)

    # # Plot labels prediction base on the defined model
    # plot_functions.plot_test_set_prediction(model=model, model_name=model_name, test_dataset=test_dataset,
    #                                         show_on_screen=False, store_in_folder=False)

# DA FARE
# Definire funzione da richiamare nel main con tutti i modelli che compila e stampa valutazione

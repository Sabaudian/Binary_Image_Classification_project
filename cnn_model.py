# Import
import keras
import tensorflow as tf
import keras_tuner as kt

from keras import layers

# My import
import plot_functions
import constants as const


# ******************************************************* #
# ********* DEFINITION AND TUNING OF THE MODELS ********* #
# ******************************************************* #

# Build my CNN model
def define_cnn_model(hp):
    my_cnn_model = tf.keras.Sequential()

    my_cnn_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                                   input_shape=const.INPUT_SHAPE, name="convolution_1"))
    my_cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1"))
    my_cnn_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", name="convolution_2"))
    my_cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2"))
    my_cnn_model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu", name="convolution_3"))
    my_cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    my_cnn_model.add(layers.Flatten(name="flatten"))

    hps_units = hp.Int("units", min_value=32, max_value=512, step=32)

    my_cnn_model.add(layers.Dense(units=hps_units, activation="relu", name="hidden_layer"))
    my_cnn_model.add(layers.Dense(units=1, activation="sigmoid", name="output_layer"))

    my_cnn_model.summary()
    keras.utils.plot_model(model=my_cnn_model, to_file="plot/my_cnn_tuned_model.png", dpi=300)

    hps_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    my_cnn_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=hps_learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return my_cnn_model


# Perform hyperparameter Tuning
def tuning_cnn(train_data, validation_data, test_data):
    tuner_cnn = kt.Hyperband(
        hypermodel=define_cnn_model,
        objective="val_accuracy",
        max_epochs=5,
        factor=2,
        overwrite=True,
        directory="models",
        project_name="cnn"
    )

    # Print search summary
    tuner_cnn.search_space_summary()
    # Stop training when a monitored metric (i.e.: val_loss) has stopped improving.
    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    # Performs a search for best hyperparameter configurations.
    tuner_cnn.search(train_data, epochs=10, validation_data=validation_data, callbacks=[stop_early])
    # Retrieve the best hyperparameters
    best_hps = tuner_cnn.get_best_hyperparameters(num_trials=1)[0]

    print("\n> The hyperparameter search is complete!"
          "\n- The optimal number of units in the first densely-connected layer is: {}"
          .format(best_hps.get("units")) +
          "\n- The optimal learning rate for the optimizer is: {}\n".format(best_hps.get("learning_rate")))

    # Build the model with the optimal hyperparameters and train it on the data for 10 epochs
    print("\n> Build the model with the optimal hyperparameters and train it on the data for 10 epochs\n")
    model = tuner_cnn.hypermodel.build(best_hps)
    history = model.fit(train_data, epochs=10, validation_data=validation_data)

    # Plot history before tuning
    plot_functions.plot_history(model_history=history, model_name="CNN", show_on_screen=True,
                                store_in_folder=True)

    # Compute best epoch value
    val_accuracy_per_epochs = history.history["val_accuracy"]
    best_epoch = val_accuracy_per_epochs.index(max(val_accuracy_per_epochs)) + 1
    print("\n> Best Epoch: {}\n".format(best_epoch, ))

    # Rerun the model with the optimal value for the epoch
    hypermodel_cnn = tuner_cnn.hypermodel.build(best_hps)
    # Retrain the model
    history_cnn = hypermodel_cnn.fit(train_data, epochs=best_epoch, validation_data=validation_data)

    # Plot history after tuning
    plot_functions.plot_history(model_history=history_cnn, model_name="tuned_CNN", show_on_screen=True,
                                store_in_folder=True)

    # Evaluate the tuned model
    test_loss, test_accuracy = hypermodel_cnn.evaluate(test_data)
    print("\n> Test Loss: {}".format(test_loss))
    print("\n> Test Accuracy: {}\n".format(test_accuracy))

    # Plot labels prediction base on the defined model
    plot_functions.plot_prediction(model=hypermodel_cnn, model_name="CNN", test_data=test_data, show_on_screen=True,
                                   store_in_folder=True)


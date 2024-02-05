# Import
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# My import
import constants as const
import utils.general_functions as general


# ************************************** #
# *********** PLOT FUNCTIONS *********** #
# ************************************** #

def show_and_save_plot(show, save, plot_folder, plot_name, plot_extension, dpi=96):
    """
    Manage the display and saving of a plot.

    :param show: If True, display the plot.
    :param save: If True, save the plot.
    :param plot_folder: The directory where the plot will be saved.
    :param plot_name: The name of the plot file (excluding the extension).
    :param plot_extension: The file extension of the plot (e.g., 'png', 'jpg').
    :param dpi: Dots per inch (resolution) for the saved image.
                Default is 96.
    """
    if show and save:  # show and store plot
        general.makedir(plot_folder)
        plt.savefig(os.path.join(plot_folder, plot_name + plot_extension), dpi=dpi)
        plt.show()
    elif show and not save:  # show plot
        plt.show()
    elif save and not show:  # store plot
        general.makedir(plot_folder)
        plt.savefig(os.path.join(plot_folder, plot_name + plot_extension), dpi=dpi)
        plt.close()
    else:  # do not show or save
        plt.close()


# Show the amount of data per class
def plot_class_distribution(train_data, test_data, show_on_screen=True, store_in_folder=True):
    """
    Plot a histogram showing the distribution of data per class in the dataset.

    :param train_data: Pandas.DataFrame.
        The training data as a Pandas DataFrame.
    :param test_data: Pandas.DataFrame.
        The test data as a Pandas DataFrame.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Defaults to True.

    Notes:
        The histogram illustrates the number of images for each class in both the training and test sets.

    :return: None
    """
    # plot dataframe, counting data in it
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle("DATA VISUALIZATION CHART", fontsize=18, weight="bold")
    sns.countplot(data=train_data, x="label", ax=ax1, hue="label", edgecolor="black")
    sns.countplot(data=test_data, x="label", ax=ax2, hue="label", edgecolor="black")

    # settings first plot
    ax1.set_title("Training Set", fontsize=16, weight="bold")
    ax1.set_xlabel(xlabel="Label", fontsize=14)
    ax1.set_ylabel(ylabel="Count", fontsize=14)

    # settings second plot
    ax2.set_title("Test Set", fontsize=16, weight="bold")
    ax2.set_xlabel(xlabel="Label", fontsize=14)
    ax2.set_ylabel(ylabel="Count", fontsize=14)

    # plot the exact amount of train's data
    for data in ax1.patches:
        x = data.get_x() + data.get_width() / 2  # text centered
        y = data.get_y() + data.get_height()  # text placed at column height => number of images in that label
        value = int(data.get_height())  # get value
        ax1.text(x, y, value, ha="center", fontsize=14, weight="bold")

    # plot the exact amount of test's data
    for data in ax2.patches:
        x = data.get_x() + data.get_width() / 2  # text centered
        y = data.get_y() + data.get_height()  # text placed at column height => number of images in that label
        value = int(data.get_height())  # get value
        ax2.text(x, y, value, ha="center", fontsize=14, weight="bold")

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="class_distribution_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot some images with the corresponding class label
def plot_view_dataset(train_ds, show_on_screen=True, store_in_folder=True):
    """
    Visualize a subset of images from the dataset.

    :param train_ds: tf.data.Dataset
        A TensorFlow dataset object corresponding to the training data.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Defaults to True.

    :notes:
        The function plots a 3x3 grid of images from the dataset's first batch, along with their corresponding labels.
        The images are displayed with labels, and the plot can be shown on screen and/or saved to a specified folder.

    :returns: None
    """
    # Plot
    plt.figure(figsize=(16, 8))

    # Add a title to the entire plot
    plt.suptitle(t="Dataset Snapshot: Visualizing Images with Class Labels", fontsize=18, weight="bold")

    # Take the first batch of images and labels from the dataset
    for images, labels in train_ds.take(1):
        # Loop through the first 9 images in the batch
        for i in range(9):
            # Create subplots in a 3x3 grid
            plt.subplot(3, 3, i + 1)

            # Display the image
            plt.imshow(images[i].numpy().astype("uint8"))

            # Determine the class label based on the dataset
            class_label = "chihuahua" if labels[i] == 0 else "muffin"

            # Add title with label information
            plt.title(label=f"{class_label}", fontsize=16)
            # Adjust layout and turn off axis for cleaner presentation
            plt.tight_layout()
            plt.axis("off")

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="show_images_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot augmented images
def plot_data_augmentation(train_ds, data_augmentation, show_on_screen=True, store_in_folder=True):
    """
    Plot augmented images generated from a data augmentation pipeline.

    :param train_ds: The training dataset contains original images.
    :param data_augmentation: The data augmentation pipeline applied to the images.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Defaults to True.
    """

    # Plot
    plt.figure(figsize=(10, 8))

    # Add a title to the entire plot
    plt.suptitle(t="Data Augmentation Example", fontsize=22, weight="bold")

    # Take the first batch of images from the dataset
    for images, _ in train_ds.take(1):
        # Loop through the first 9 images in the batch
        for i in range(9):
            # Apply data augmentation
            augmented_images = data_augmentation(images)

            # Create subplots in a 3x3 grid
            plt.subplot(3, 3, i + 1)

            # Display the image
            plt.imshow(augmented_images[0].numpy().astype("uint8"))

            plt.axis("off")

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="data_augmentation_plot",
        plot_extension=const.FILE_EXTENSION
    )


def plot_history(history, model_name, show_on_screen=True, store_in_folder=True):
    """
    Visualize the training history of the model.

    :param history: The training history of the model (e.g., history = model.fit()).
    :param model_name: The name of the model for labeling the plot.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Defaults to True.
    """
    # Plot
    plt.figure(figsize=(16, 8))

    # Add a title to the entire plot
    plt.suptitle("{} Training History".format(model_name), fontsize=18)

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], linewidth=3)
    plt.plot(history.history["val_accuracy"], linewidth=3)
    plt.title(label="Training and Validation Accuracy", fontsize=16)
    plt.ylabel(ylabel="accuracy", fontsize=14)
    plt.xlabel(xlabel="epoch", fontsize=14)
    plt.grid()
    plt.legend(["Train", "Validation"], loc="best")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], linewidth=3)
    plt.plot(history.history["val_loss"], linewidth=3)
    plt.title(label="Training and Validation Loss", fontsize=16)
    plt.ylabel(ylabel="Loss", fontsize=14)
    plt.xlabel(xlabel="epoch", fontsize=14)
    plt.grid()
    plt.legend(["Train", "Validation"], loc="best")

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_training_history_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot confusion matrix
def plot_confusion_matrix(model, model_name, x_test, y_test, show_on_screen=True, store_in_folder=True):
    """
    Plot the Confusion Matrix.

    :param model: The model.
    :param model_name: Name assigned to the model.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Defaults to True.
    """
    # Predict
    predict = model.predict(x=x_test, verbose=0)

    # Convert the predictions to binary classes (0 or 1)
    y_pred = (predict >= 0.5).astype("int32")

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(model_name + " Confusion Matrix", fontsize=18)
    ax.set_xlabel(xlabel="Predicted Label", fontsize=16)
    ax.set_ylabel(ylabel="True Label", fontsize=16)
    ax.tick_params(labelsize=12)

    # Compute the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix as a heatmap
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Chihuahua", "Muffin"])
    display.plot(cmap="viridis", values_format="d", ax=ax)

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_confusion_matrix_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Bar graph: Real values vs Predict values
def plot_model_predictions_evaluation(model, model_name, class_list, x_test, y_test, show_on_screen=True,
                                      store_in_folder=True):
    """
    Plot a bar graph that compares true classes with the ones predicted by the model in input.

    :param model: tensorflow.keras.Model
        The input model for predictions.
    :param model_name: str
        The name of the model.
    :param class_list: list of str
        The list of class names.
    :param x_test: numpy.ndarray
        Input values of the test dataset.
    :param y_test: numpy.ndarray
        Target values of the test dataset.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Default is True.
    """
    # Predict
    predict = model.predict(x=x_test, verbose=0)
    # Convert the predictions to binary classes (0 or 1)
    y_pred = (predict >= 0.5).astype("int32")

    # Classes
    classes = {i: const.CLASS_LIST[i] for i in range(0, len(const.CLASS_LIST))}

    # Create a DataFrame for storing class comparison data
    clf_data = pd.DataFrame(columns=["real_class_num", "predict_class_num",
                                     "real_class_label", "predict_class_label"])
    clf_data["real_class_num"] = y_test
    clf_data["predict_class_num"] = y_pred

    # Compare True classes with predicted ones
    comparison_column = np.where(clf_data["real_class_num"] == clf_data["predict_class_num"], True, False)
    clf_data["check"] = comparison_column

    clf_data["real_class_label"] = clf_data["real_class_num"].replace(classes)
    clf_data["predict_class_label"] = clf_data["predict_class_num"].replace(classes)

    # Create a DataFrame for input data and count
    input_data = pd.DataFrame()
    input_data[["Images", "Real_Value"]] = \
        clf_data[["real_class_label", "predict_class_label"]].groupby(["real_class_label"], as_index=False).count()
    input_data[["Images", "Predict_Value"]] = \
        clf_data[["real_class_label", "predict_class_label"]].groupby(["predict_class_label"], as_index=False).count()

    # Plot
    ax = input_data.plot(kind="bar", figsize=(16, 8), fontsize=12,
                         width=0.6, color={"#006400", "#ffd700"}, edgecolor="black")

    ax.set_xticklabels(class_list, rotation=0)
    ax.legend(["Real Value", "Predict Value"], fontsize=9, loc="upper right")
    plt.title(label=model_name + " Predictions Evaluation", fontsize=18)
    plt.xlabel(xlabel="Classes", fontsize=16)
    plt.ylabel(ylabel="Occurrences", fontsize=16)

    # Annotate each bar with its height, i.e., the Real Value vs the Predicted one
    for p in ax.patches:
        ax.annotate(format(p.get_height()),
                    (p.get_x() + (p.get_width() / 2), p.get_height()), ha="center", va="center",
                    xytext=(0, 5), textcoords="offset points", fontsize=14, rotation=0)

    # Show and/or store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_prediction_evaluation_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot test images with prediction
def plot_visual_prediction(model, model_name, x_test, y_test, randomize=False, show_on_screen=True, store_in_folder=True):
    """
    Plots a visual representation of the model predictions on a test dataset.

    :param model: tensorflow.keras.Model
        The trained model for making predictions.
    :param model_name: str
        A string representing the name of the model.
    :param x_test: numpy.ndarray
        Input test data (images).
    :param y_test: numpy.ndarray
        True labels for the test data.
    :param randomize: If True, pick random images.
        Default is False.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Default is True.
    """

    if randomize:

        # Select random indices from x_test
        num_samples = min(9, len(x_test))  # Adjust the number of samples to display
        random_indices = random.sample(range(len(x_test)), num_samples)

        # Select a subset of x_test and y_test based on the random indices
        x_test = x_test[random_indices]
        y_test = y_test[random_indices]

    # Predict
    predicts = model.predict(x=x_test, verbose=0)

    # Convert the predictions to binary classes (0 or 1)
    predicted_classes = (predicts >= 0.5).astype("int32")

    # Assign class name to class indices (chihuahua = 0, muffin = 1)
    predicted_class_labels = ["chihuahua" if pred_label == 0 else "muffin" for pred_label in predicted_classes]

    # Plot configuration
    figure_size = (16, 8)
    subplot_rows, subplot_cols = 3, 3

    # Plot
    plt.figure(figsize=figure_size)

    # Add a title to the entire plot
    plt.suptitle("{} Visual Prediction".format(model_name), fontsize=18, weight="bold")

    for i in range(min(subplot_rows * subplot_cols, len(x_test))):
        plt.subplot(subplot_rows, subplot_cols, i + 1)

        # Extract a single image from the test data using the index
        single_image = x_test[i]
        plt.imshow(single_image)

        # True class indices, assuming 0 represents "chihuahua" and 1 represents "muffin"
        true_class_label = "chihuahua" if y_test[i] == 0 else "muffin"

        # Add Visual aid to check the correctness of the prediction
        title_color = "green" if true_class_label == predicted_class_labels[i] else "red"

        # Add a title to all images (true vs predicted class label) with a background for a better presentation
        plt.title(
            label="TRUE: {}\nPREDICTED: {}".format(true_class_label, predicted_class_labels[i]),
            fontsize=10, color=title_color,
            bbox=dict(facecolor="lightgray", alpha=0.7, edgecolor="black", boxstyle="round,pad=0.5"),
            fontweight="bold"
        )
        plt.tight_layout()
        plt.axis("off")

    # Show and/or store the plot
    if randomize:

        show_and_save_plot(
            show=show_on_screen, save=store_in_folder,
            plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
            plot_name=model_name + "_visual_random_prediction_plot",
            plot_extension=const.FILE_EXTENSION
        )

    else:

        show_and_save_plot(
            show=show_on_screen, save=store_in_folder,
            plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
            plot_name=model_name + "_visual_prediction_plot",
            plot_extension=const.FILE_EXTENSION
        )


# Plot the training history for each fold in kfold cross validation
def plot_fold_history(fold_history, model_name, show_on_screen=True, store_in_folder=True):
    """
    Visualize the training history of a model during KFold cross-validation

    :param fold_history: The training history of the model (e.g., history = model.fit()), collected during KFold.
    :param model_name: The name of the model for labeling the plot.
    :param show_on_screen: If True, display the plot on the screen.
        Default is True.
    :param store_in_folder: If True, save the plot in a specified folder.
        Default is True.
    """
    # Plot the training history for each fold
    for fold in range(len(fold_history)):
        # Plot size
        plt.figure(figsize=(16, 8))

        # Add a title to the entire plot
        plt.suptitle("{} Fold {} Training History".format(model_name, fold + 1), fontsize=18)

        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(fold_history[fold].history["accuracy"], linewidth=3)
        plt.plot(fold_history[fold].history["val_accuracy"], linewidth=3)
        plt.title(label="Training and Validation Accuracy", fontsize=16)
        plt.ylabel(ylabel="accuracy", fontsize=14)
        plt.xlabel(xlabel="epoch", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.legend(["Train", "Validation"], loc="upper right")

        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(fold_history[fold].history["loss"], linewidth=3)
        plt.plot(fold_history[fold].history["val_loss"], linewidth=3)
        plt.title(label="Training and Validation Loss", fontsize=16)
        plt.ylabel(ylabel="Loss", fontsize=14)
        plt.xlabel(xlabel="epoch", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.legend(["Train", "Validation"], loc="upper right")

        # Zero-one Loss
        plt.subplot(1, 3, 3)
        plt.plot(fold_history[fold].history["zero_one_loss"], linewidth=3)
        plt.plot(fold_history[fold].history["val_zero_one_loss"], linewidth=3)
        plt.title(label="Training and Validation Zero-one Loss", fontsize=16)
        plt.ylabel(ylabel="Zero-one Loss", fontsize=14)
        plt.xlabel(xlabel="epoch", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.legend(["Train", "Validation"], loc="upper right")

        # Show and store the plot
        show_and_save_plot(
            show=show_on_screen, save=store_in_folder,
            plot_folder=os.path.join(const.PLOT_FOLDER, "KFold", model_name),
            plot_name=model_name + "_fold_" + f"{fold + 1}_training_history_plot",
            plot_extension=const.FILE_EXTENSION
        )

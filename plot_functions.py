# Import
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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
def plot_img_class_histogram(train_data, test_data, show_on_screen=True, store_in_folder=True):
    """
    Plot a histogram that shows the amount of data per class of the dataset.
    The histogram shows the number of images that represent muffin and chihuahua for the training and test set.

    :param train_data: Pandas.Dataframe, training data.
    :param test_data: Pandas.Dataframe, test data.
    :param show_on_screen: Boolean value, if True, shows the plot.
    :param store_in_folder: Boolean value, if True, saves the plot.
    """

    # plot dataframe, counting data in it
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle("DATA VISUALIZATION CHART", fontsize=18, weight="bold")
    sns.countplot(data=train_data, x="label", ax=ax1, hue="label", edgecolor="black")
    sns.countplot(data=test_data, x="label", ax=ax2, hue="label", edgecolor="black")

    # settings first plot
    ax1.set_title("Train Set", fontsize=16, weight="bold")
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
        ax1.text(x, y, value, ha="center", fontsize=12, weight="bold")

    # plot the exact amount of test's data
    for data in ax2.patches:
        x = data.get_x() + data.get_width() / 2  # text centered
        y = data.get_y() + data.get_height()  # text placed at column height => number of images in that label
        value = int(data.get_height())  # get value
        ax2.text(x, y, value, ha="center", fontsize=12, weight="bold")

    # store plot into a proper folder
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="dataset_histogram_overview_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot the value's distribution of the images in the dataset
def plot_img_aspect_ratio(image_metadata, show_on_screen=True, store_in_folder=True):
    """
    Value's distribution of the images in the dataset

    :param image_metadata: Metadata information collected from images in the dataset.
    :param show_on_screen: Boolean value, if True, shows the plot.
    :param store_in_folder: Boolean value, if True, saves the plot.
    """
    # Initialize value
    width, height, label = image_metadata["width"], image_metadata["height"], image_metadata["label"]

    # plot the image
    plt.subplots(figsize=(16, 8))
    sns.scatterplot(x=width, y=height, hue=label)

    # Show and store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="img_aspect_ratio_plot",
        plot_extension=const.FILE_EXTENSION
    )


def plot_view_dataset(train_ds, show_on_screen=True, store_in_folder=True):
    """
    View some images that composed the dataset.

    :param train_ds: tf.Dataset.data object corresponding to the train data.
    :param show_on_screen: Boolean value, if True, shows the plot.
    :param store_in_folder: Boolean value, if True, saves the plot.
    """
    plt.figure(figsize=(16, 8))
    for images, labels in train_ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if labels[i] == 0:
                plt.title("chihuahua")
            else:
                plt.title("muffin")
            plt.axis("off")

    # Show and store the plot
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
    :param store_in_folder: If True, save the plot in a folder.
        Default is True.
    """

    # Plot
    plt.figure(figsize=(16, 8))

    # Add a title to the entire plot
    plt.suptitle("Data Augmentation", fontsize=22, weight="bold")

    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    # Show and store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=const.PLOT_FOLDER,
        plot_name="data_augmentation_plot",
        plot_extension=const.FILE_EXTENSION
    )


def plot_history(history, model_name, show_on_screen=True, store_in_folder=True):
    """
    Visualize the training history of a model.

    :param history: The training history of the model (e.g., history = model.fit()).
    :param model_name: The name of the model for labeling the plot.
    :param show_on_screen: If True, display the plot on the screen.
    Default is True.
    :param store_in_folder: If True, save the plot in a folder.
    Default is True.

    Examples:
    - To plot the training history with default options:
      plot_history(my_model_history, "MyModel")

    - To plot the history and only display on screen:
      plot_history(my_model_history, "MyModel", show_on_screen=True, store_in_folder=False)

    - To plot the history and save the plot without displaying:
      plot_history(my_model_history, "MyModel", show_on_screen=False, store_in_folder=True)
    """
    # Plot
    plt.figure(figsize=(16, 8))

    # Add a title to the entire plot
    plt.suptitle("{} Training History".format(model_name), fontsize=18)

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], linewidth=3)
    plt.plot(history.history["val_accuracy"], linewidth=3)
    plt.title(label="Training and Validation accuracy", fontsize=16)
    plt.ylabel(ylabel="accuracy", fontsize=14)
    plt.xlabel(xlabel="epoch", fontsize=14)
    plt.grid()
    plt.legend(["Train", "Validation"], loc="best")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], linewidth=3)
    plt.plot(history.history["val_loss"], linewidth=3)
    plt.title(label="Training and Validation  Loss", fontsize=16)
    plt.ylabel(ylabel="Loss", fontsize=14)
    plt.xlabel(xlabel="epoch", fontsize=14)
    plt.grid()
    plt.legend(["Train", "Validation"], loc="best")

    # Show and store the plot
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
    :param show_on_screen: Boolean value, if True, shows the plot.
    :param store_in_folder: Boolean value, if True, saves the plot.
    """

    # Predict
    predicts = model.predict(x_test)
    # Convert the predictions to binary classes (0 or 1)
    predicted_classes = (predicts >= 0.5).astype(int)
    predicted_classes = predicted_classes.flatten()

    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(model_name + " Confusion Matrix", fontsize=18)
    ax.set_xlabel(xlabel="Predicted Label", fontsize=16)
    ax.set_ylabel(ylabel="True Label", fontsize=16)
    ax.tick_params(labelsize=12)

    # Compute the confusion matrix
    confusion = confusion_matrix(y_test, predicted_classes)

    # Display the confusion matrix as a heatmap
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Chihuahua", "Muffin"])
    display.plot(cmap="viridis", values_format="d", ax=ax)

    # Show and store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_confusion_matrix_plot",
        plot_extension=const.FILE_EXTENSION
    )


def plot_predictions_evaluation(model, model_name, class_list, x_test, y_test, show_on_screen=True,
                                store_in_folder=True):
    """
    Plot a bar graph that compares true classes with the one predicted by the model in input.
    :param model: Model in input.
    :param model_name: Model name.
    :param class_list: The list of class names.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    :param show_on_screen: Boolean value, if True, shows the plot.
    :param store_in_folder: Boolean value, if True, saves the plot.
    """
    # Predict
    predicts = model.predict(x_test)
    # Convert the predictions to binary classes (0 or 1)
    predicted_classes = (predicts >= 0.5).astype(int)
    predicted_classes = predicted_classes.flatten()

    # Classes
    classes = {i: const.CLASS_LIST[i] for i in range(0, len(const.CLASS_LIST))}

    clf_data = pd.DataFrame(columns=["real_class_num", "predict_class_num",
                                     "real_class_label", "predict_class_label"])
    clf_data["real_class_num"] = y_test
    clf_data["predict_class_num"] = predicted_classes

    # Compare True classes with predicted ones
    comparison_column = np.where(clf_data["real_class_num"] == clf_data["predict_class_num"], True, False)
    clf_data["check"] = comparison_column

    clf_data["real_class_label"] = clf_data["real_class_num"].replace(classes)
    clf_data["predict_class_label"] = clf_data["predict_class_num"].replace(classes)

    input_data = pd.DataFrame()
    input_data[["Genre", "Real_Value"]] = \
        clf_data[["real_class_label", "predict_class_label"]].groupby(["real_class_label"], as_index=False).count()
    input_data[["Genre", "Predict_Value"]] = \
        clf_data[["real_class_label", "predict_class_label"]].groupby(["predict_class_label"], as_index=False).count()

    # Plot
    ax = input_data.plot(kind="bar", figsize=(16, 8), fontsize=12,
                         width=0.6, color={"#006400", "#ffd700"}, edgecolor="black")

    ax.set_xticklabels(class_list, rotation=0)
    ax.legend(["Real Value", "Predict Value"], fontsize=9, loc="upper right")
    plt.title(label=model_name + " Predictions Evaluation", fontsize=18)
    plt.xlabel(xlabel="Classes", fontsize=16)
    plt.ylabel(ylabel="Occurrences", fontsize=16)

    for p in ax.patches:
        ax.annotate(format(p.get_height()),
                    (p.get_x() + (p.get_width() / 2), p.get_height()), ha="center", va="center",
                    xytext=(0, 5), textcoords="offset points", fontsize=12, rotation=0)

    # Show and store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_prediction_evaluation_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Plot test images with prediction
def plot_visual_prediction(model, model_name, x_test, y_test, show_on_screen=True, store_in_folder=True):
    """
    Plots a visual representation of the model predictions on a test dataset.
    This function generates a 3x3 grid of randomly selected test images along with their true and predicted labels.
    The predicted labels are obtained by thresholding the model's output probabilities at 0.5.
    The plot is displayed on the screen and/or saved in a specified folder.

    :param model: The trained model for making predictions.
    :param model_name: A string representing the name of the model.
    :param x_test: Input test data (images).
    :param y_test: True labels for the test data.
    :param show_on_screen: Boolean, whether to display the plot on the screen (default=True).
    :param store_in_folder: Boolean, whether to save the plot in a folder (default=True).
    """
    # Predict
    predicts = model.predict(x_test)
    # Convert the predictions to binary classes (0 or 1)
    predicted_classes = (predicts >= 0.5).astype(int)
    predicted_classes = predicted_classes.flatten()

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

        # Add a title to every image with a background
        plt.title(
            label="TRUE: {}\nPREDICTED: {}".format(true_class_label, predicted_class_labels[i]),
            fontsize=10, color=title_color,
            bbox=dict(facecolor="lightgray", alpha=0.7, edgecolor="black", boxstyle="round,pad=0.5"),
            fontweight="bold"
        )
        plt.tight_layout()
        plt.axis("off")

    # Show and store the plot
    show_and_save_plot(
        show=show_on_screen, save=store_in_folder,
        plot_folder=os.path.join(const.PLOT_FOLDER, model_name),
        plot_name=model_name + "_visual_prediction_plot",
        plot_extension=const.FILE_EXTENSION
    )


# Posizionare testo ecc.
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/titles_demo.html

# FUNZIONI DA DEFINIRE
# https://towardsdatascience.com/10-minutes-to-building-a-binary-image-classifier-by-applying-transfer-learning-to-mobilenet-eab5a8719525
# def plot_roc_curve(false_positive_rate, true_positive_rate, validation_data):


# # Plot test images with prediction
# def plot_visual_prediction(model, model_name, test_dataset, show_on_screen=True, store_in_folder=True):
#     # Predict
#     predicts = model.predict(test_dataset)
#     # Convert the predictions to class indices (0 or 1)
#     predicted_classes = (predicts >= 0.5).astype("int")
#
#     # Assign class name to class indices (chihuahua = 0, muffin = 1)
#     predicted_class_labels = ["chihuahua" if pred_label == 0 else "muffin" for pred_label in predicted_classes]
#
#     # Plot configuration
#     figure_size = (16, 8)
#     subplot_rows, subplot_cols = 3, 3
#
#     # Plot
#     plt.figure(figsize=figure_size)
#
#     # Add a title to the entire plot
#     plt.suptitle("{} Visual Prediction".format(model_name), fontsize=18, weight="bold")
#
#     # Shuffle the entire test dataset
#     num_elements_in_test_data = tf.data.experimental.cardinality(test_dataset).numpy()
#     shuffled_test_data = test_dataset.shuffle(buffer_size=num_elements_in_test_data, reshuffle_each_iteration=False)
#     batch = next(iter(shuffled_test_data.batch(32)))
#
#     # Randomly select 9 indices from the batch
#     random_indices = random.sample(range(32), k=9)
#
#     for i, idx in enumerate(random_indices):
#         plt.subplot(subplot_rows, subplot_cols, i + 1)
#
#         # Extract a single image from the batch using the random index
#         single_image = batch[0][idx].numpy()[0].astype("uint8")
#         plt.imshow(single_image)
#
#         # True class indices, 0 represents "chihuahua" and 1 represents "muffin"
#         true_class_labels = ["chihuahua" if idx == 0 else "muffin" for idx in batch[1][idx]]
#
#         # Add Visual aid to check the correctness of the prediction
#         title_color = "green" if true_class_labels[idx] == predicted_class_labels[idx] else "red"
#
#         # Add a title to every image with a background
#         plt.title(
#             label="TRUE: {}\nPREDICTED: {}".format(true_class_labels[idx], predicted_class_labels[idx]),
#             fontsize=10, color=title_color,
#             bbox=dict(facecolor="lightgray", alpha=0.7, edgecolor="black", boxstyle="round,pad=0.5"),
#             fontweight="bold"
#         )
#         plt.tight_layout()
#         plt.axis("off")
#
#     # Show and store the plot
#     show_and_save_plot(
#         show=show_on_screen, save=store_in_folder,
#         plot_folder=const.PLOT_FOLDER,
#         plot_name=model_name + "_visual_prediction_plot",
#         plot_extension=const.FILE_EXTENSION
#     )

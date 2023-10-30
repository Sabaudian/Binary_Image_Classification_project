# Import
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# My import
import constants as const
import utils.general_functions as general


# ************************************** #
# *********** PLOT FUNCTIONS *********** #
# ************************************** #


# Show the amount of data per class
def plot_img_class_histogram(train_data, test_data, show_on_screen=True, store_in_folder=True):
    """
    Plot a histogram that shows the amount of data per class of the dataset.
    The histogram shows the number of images that represent muffin and chihuahua for the training and test set.
    :param train_data: Pandas.Dataframe, training data
    :param test_data: Pandas.Dataframe, test data
    :param show_on_screen: boolean, decides if show plot on screen
    :param store_in_folder: boolean, decides if save plot on location
    """

    # plot dataframe, counting data in it
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle("DATA VISUALIZATION CHART", fontsize=18, weight="bold")
    sns.countplot(data=train_data, x="label", ax=ax1, hue="label")
    sns.countplot(data=test_data, x="label", ax=ax2, hue="label")

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
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_dataset_visualization" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()

    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()
    else:
        plt.close()


# plot the value's distribution of the images in the dataset
def plot_img_aspect_ratio(image_metadata, show_on_screen=True, store_in_folder=True):
    """
    Value's distribution of the images in the dataset
    :param image_metadata:
    :param show_on_screen:
    :param store_in_folder:
    """
    # Initialize value
    width, height, label = image_metadata["width"], image_metadata["height"], image_metadata["label"]

    # plot the image
    plt.subplots(figsize=(16, 8))
    sns.scatterplot(x=width, y=height, hue=label)

    # choose if store image to a path
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_img_aspect_ratio" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()

    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()
    else:
        plt.close()


def plot_view_dataset(train_ds, show_on_screen=True, store_in_folder=True):
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

    # if it is true, save the image
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_show_images" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()

    # if it is true, show the image
    if show_on_screen:
        plt.show()
    else:
        plt.close()


def plot_data_augmentation(train_ds, data_augmentation, show_on_screen=True, store_in_folder=True):
    plt.figure(figsize=(16, 8))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_data_augmentation" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()
    else:
        plt.close()


def plot_history(history, model_name, show_on_screen=True, store_in_folder=True):
    # define image dimension
    plt.figure(figsize=(16, 8))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], linewidth=3)
    plt.plot(history.history["val_accuracy"], linewidth=3)
    plt.title("Training and Validation Accuracy", fontsize=18)
    plt.ylabel("accuracy", fontsize=16)
    plt.xlabel("epoch", fontsize=16)
    plt.grid()
    plt.legend(["Training", "Validation"], loc="best")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], linewidth=3)
    plt.plot(history.history["val_loss"], linewidth=3)
    plt.title("Training and Validation Loss", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("epoch", fontsize=16)
    plt.grid()
    plt.legend(["Training", "Validation"], loc="best")

    # to store plot
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "_history" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()
    else:
        plt.close()


# plot confusion matrix to evaluate the model
def plot_confusion_matrix(model, model_name, x_test, y_test, show_on_screen=True, store_in_folder=True):
    """
    Plot the Confusion Matrix.
    :param model: The model.
    :param model_name: Name assigned to the model.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    :param show_on_screen: Boolean, decides to show plot on screen or not
    :param store_in_folder: boolean, decides to save plot or not
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

    # to store plot
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "_confusion_matrix" + const.FILE_EXTENSION, dpi=300)
        if not show_on_screen:
            plt.close()
    # show on screen
    if show_on_screen:
        # plot the image
        plt.show()


# # da testare
# def plot_error_analysis(model, test_dataset, test_ds):
#     val_pred = model.predict(test_ds, steps=np.ceil(test_dataset.shape[0] / 32))
#     test_dataset.loc[:, "val_pred"] = np.argmax(val_pred, axis=1)
#
#     labels = dict((v, k) for k, v in test_ds.class_indices.items())
#
#     test_dataset.loc[:, "val_pred"] = test_dataset.loc[:, "val_pred"].map(labels)
#
#     test_errors = test_dataset[test_dataset.label != test_dataset.val_pred].reset_index(drop=True)


def plot_test_set_prediction(model, model_name, test_dataset, show_on_screen=True, store_in_folder=True):
    """
    PLot a visual representation of the labels prediction based on the input model,
    and compare them with the true labels
    :param model: The model in input
    :param model_name: The model name
    :param test_dataset: Raw test dataset
    :param show_on_screen: Boolean, decides if show plot on screen
    :param store_in_folder: Boolean, decides if show plot on screen
    """
    # Predict
    label_prediction = np.argmax(model.predict(test_dataset), axis=-1)
    predicted_class_labels = ["chihuahua" if pred_label == 0 else "muffin" for pred_label in label_prediction]

    # Plot
    plt.figure(figsize=(16, 8))
    for images, labels in test_dataset.take(1):
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title("True Label: {}\n Predicted: {}\n".format(labels[i], predicted_class_labels[i]), fontsize=7)
            plt.axis("off")

    # if it is true, save the image
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "_labels_prediction" + const.FILE_EXTENSION,
                    dpi=300)
        if not show_on_screen:
            plt.close()
    # if it is true, show the image
    if show_on_screen:
        plt.tight_layout()
        plt.show()


def plot_predictions_evaluation(input_data, model_name, class_list, show_on_screen=True, store_in_folder=True):

    ax = input_data.plot(kind="bar", figsize=(16, 8), fontsize=14,
                         width=0.6, color={"#006400", "#ffd700"}, edgecolor="black")

    ax.set_xticklabels(class_list, rotation=0)
    ax.legend(["Real Value", "Predict Value"], fontsize=9, loc="upper right")
    plt.title("Predictions Evaluation - " + model_name.upper(), fontsize=22)
    plt.xlabel("Genres", fontsize=18)
    plt.ylabel("Occurrences", fontsize=18)

    for p in ax.patches:
        ax.annotate(format(p.get_height()),
                    (p.get_x() + (p.get_width() / 2), p.get_height()), ha="center", va="center",
                    xytext=(0, 5), textcoords="offset points", fontsize=10, rotation=0)

    # if it is true, save the image
    if store_in_folder:
        general.makedir(const.PLOT_FOLDER)
        plt.savefig(const.PLOT_FOLDER + "/plot_" + model_name + "prediction_evaluation" + const.FILE_EXTENSION,
                    dpi=300)
        if not show_on_screen:
            plt.close()
    # if it is true, show the image
    if show_on_screen:
        plt.tight_layout()
        plt.show()

# Posizionare testo ecc.
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/titles_demo.html

# FUNZIONI DA DEFINIRE
# https://towardsdatascience.com/10-minutes-to-building-a-binary-image-classifier-by-applying-transfer-learning-to-mobilenet-eab5a8719525
# def plot_roc_curve(false_positive_rate, true_positive_rate, validation_data):

import seaborn as sns
import matplotlib.pyplot as plt

# My import
import constants as const
import utils.general_functions as general


# -------------------------------------- #
# ----------- PLOT FUNCTIONS ----------- #
# -------------------------------------- #


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
    # show on screen
    if show_on_screen:
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
            plt.savefig(const.PLOT_FOLDER + "/" + const.DATASET_VISUAL_INFO + const.FILE_EXTENSION, dpi=300)

        # plot the image
        plt.show()


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
    # show on screen
    if show_on_screen:
        # plot the image
        plt.subplots(figsize=(16, 8))
        sns.scatterplot(x=width, y=height, hue=label)
        # choose if store image to a path
        if store_in_folder:
            general.makedir(const.PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/plot_img_aspect_ratio" + const.FILE_EXTENSION, dpi=300)

        # plot the image
        plt.show()


def plot_data_visualization(train_ds, show_on_screen=True, store_in_folder=True):
    class_names = train_ds.class_names
    # show on screen
    if show_on_screen:
        plt.figure(figsize=(16, 8))
        for images, labels in train_ds.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

        if store_in_folder:
            general.makedir(const.PLOT_FOLDER)
            plt.savefig(const.PLOT_FOLDER + "/plot_visual_data" + const.FILE_EXTENSION, dpi=300)
        # plot the image
        plt.show()

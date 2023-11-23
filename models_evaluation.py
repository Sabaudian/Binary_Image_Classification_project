# Import
import os
import pandas as pd

from IPython import display
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report

# My import
import plot_functions
import constants as const
import utils.general_functions as general


# ******************************************** #
# ************* MODEL EVALUATION ************* #
# ******************************************** #


# Print Test Accuracy and Test Loss
def accuracy_loss_model(model, x_test, y_test):
    """
    Compute a simple evaluation of the model,
    printing the loss and accuracy for the test set.

    :param model: Model in input.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    """
    # Compute loss and accuracy
    test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test)

    predict = model.predict(x_test)
    y_pred = (predict > 0.5).astype("int")
    test_zero_one_loss = zero_one_loss(y_test, y_pred)

    # Print evaluation info. about the model
    print("\n- Test Loss: {:.2f}%"
          "\n- Test Accuracy: {:.2f}%"
          "\n- Test Zero-one Loss: {:.2f}%\n"
          .format(test_loss * 100, test_accuracy * 100, test_zero_one_loss * 100))


def compute_evaluation_metrics(model, model_name, x_test, y_test):
    """
    Get the classification report about the model in input.
    :param model: Model in input.
    :param model_name: Name of the model.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    :return: The Classification Report Dataframe of the model
    """
    # Predict
    y_predicts = model.predict(x_test)

    # Convert the predictions to binary classes (0 or 1)
    predictions = (y_predicts > 0.5).astype("int")
    # predictions = predictions.flatten()

    # Compute report
    clf_report = classification_report(y_test, predictions, target_names=const.CLASS_LIST, digits=2, output_dict=True)

    # Update so in df is shown in the same way as standard print
    clf_report.update({"accuracy": {"precision": None, "recall": None, "f1-score": clf_report["accuracy"],
                                    "support": clf_report.get("macro avg")["support"]}})
    df = pd.DataFrame(clf_report).transpose()

    # Print report
    print("\n> Classification Report:")
    display.display(df)
    print("\n")

    # Save the report
    general.makedir(dirpath=const.DATA_PATH)
    file_path = os.path.join(const.DATA_PATH, model_name + "_classification_report.csv")
    df.to_csv(file_path, index=True, float_format="%.2f")
    return df


# Model evaluation with extrapolation of data and information plot
def evaluate_model(model, model_name, x_test, y_test):
    """
    Evaluate the Performance of the model on the test set.

    :param model: The model in input.
    :param model_name: Name of the model.
    :param x_test: Input values of the test dataset.
    :param y_test: Target values of the test dataset.
    """

    # Evaluate the model
    print("\n> " + model_name + " Model Evaluation:")

    # Compute a simple evaluation report on the model performances
    accuracy_loss_model(model=model,  x_test=x_test, y_test=y_test)

    # Compute the classification_report of the model
    compute_evaluation_metrics(model=model, model_name=model_name, x_test=x_test, y_test=y_test)

    # Plot Confusion Matrix
    plot_functions.plot_confusion_matrix(model=model, model_name=model_name, x_test=x_test, y_test=y_test,
                                         show_on_screen=True, store_in_folder=False)

    # Plot a representation of the prediction
    plot_functions.plot_model_predictions_evaluation(model=model, model_name=model_name, class_list=const.CLASS_LIST,
                                                     x_test=x_test, y_test=y_test,
                                                     show_on_screen=True, store_in_folder=False)

    # Plot a visual representation of the classification model, predicting classes
    plot_functions.plot_visual_prediction(model=model, model_name=model_name, x_test=x_test, y_test=y_test,
                                          show_on_screen=True, store_in_folder=False)

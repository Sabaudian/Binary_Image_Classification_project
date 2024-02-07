# Statistical Methods for Machine Learning: experimental project
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white">

Use Keras to train a neural network for the binary classification of muffins and Chihuahuas.

## General Information

- **Python** version is: **3.10.5**
- **Scikit-learn** version is: **1.3.1**
- **Tensorflow** version is: **2.15.0**

## How to Set up your Workspace:
Download this [dataset](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification)
and put in inside your project folder.

#### Example:
```
├── my_project_folder
│   ├── dataset
│   ├── models
│   ├── classifier.py
│   ├── constants.py
│   ├── main.py
│   └── etc.
```

> [!NOTE]
> If you want to just run it, the dataset it is not really necessary,
because through the project you can download from *Google Drive* a directory named *models*, where there are the saves of the pre-trained models and k-fold cross-validation result.

## Who ate my Chihuahua?

The project primarily utilizes the **Keras** and **Tensorflow** frameworks to set up the work environment, construct, and train neural network models. Starting with crucial preprocessing steps to assess the quality of the data, identifying corrupted files, following by a check for the possible existence of duplicate files—both factors impacting the performance of the training models. Next to this quality assessment, the data undergo a preparation step, involving transformations such as adjusting size and color format of the images. Additionally, processes like data augmentation are applied to enhance results and overall performance.

This preprocessing step sets the stage for the subsequent exploration of various network architectures. To address the task at hand, we experiment with three distinct network architectures: The **Multilayer Perceptron (MLP)**, **Convolutional Neural Network (CNN)** and **MobileNet** model. 

Beyond architectural variations, our study incorporates **hyperparameter tuning** to enhance overall performance. The resulting optimal parameters and weights are saved and utilized as the basis for **K-Fold Cross-Validation** to calculate risk estimates, employing the **zero-one loss** function. At the end, the models are evaluated on the test set to have a final understanding of their generalization capabilities.

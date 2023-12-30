# Statistical Methods for Machine Learning: experimental project
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white">

Use Keras to train a neural network for the binary classification of muffins and Chihuahuas.

## General Information

- Python version is: 3.10.5
- Scikit-learn version is: 1.3.1
- Tensorflow version is: 2.15.0

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

Artificial Neural Networks (ANNs) stand as a foundational paradigm in artificial intelligence, inspired by the intricate architecture of the human brain. In this study, we delve into the realm of image classification, specifically addressing the challenge of binary classification—discerning between images of muffins and Chihuahuas within a given data set. This task, though seemingly trivial for humans, poses a complex challenge for machines that require training to recognize nuanced differences between the two categories.

Our investigation employs the versatile Keras framework to train neural networks on transformed images. The images undergo a crucial preprocessing step, transitioning from JPG format to RGB pixel values, and are subsequently scaled down. Prior to this, meticulous preprocessing tasks such as corruption file checks and duplicate file checks were applied to ensure data integrity and eliminate redundancies within the data set.

This preprocessing step sets the stage for the subsequent exploration of various network architectures and training hyperparameters. To address the task at hand, we experiment with four distinct network architectures: The Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), MobileNet, and VGG16 models within the Keras framework serve as our tools for exploration. 

In addition to architectural variations, our study incorporates the crucial element of hyperparameter training, in order to obtain optimal values and improve overall performance. To prevent the risk of overfitting hyperparameters to the validation set, we use 5-fold cross-validation to calculate risk estimates, considering the zero-one loss for the reported cross-validated estimates. This methodology ensures an in-depth evaluation of the models' performance across different subsets of the data set, providing a comprehensive understanding of their generalization capabilities.

In addition to architectural variations, our study incorporates the crucial element of hyperparameter training, in order to obtain optimal values and improve overall performance. To prevent the risk of overfitting hyperparameters to the validation set, we use 5-fold cross-validation to calculate risk estimates, considering the zero-one loss for the reported cross-validated estimates. This methodology ensures an in-depth evaluation of the models' performance across different subsets of the dataset, providing a comprehensive understanding of their generalization capabilities.

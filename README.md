# Statistical Methods for Machine Learning: experimental project
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white">

Use Keras to train a neural network for the binary classification of muffins and Chihuahuas.

## General Information

- Python version is: 3.10.5
- Tensorflow version is: 2.14.0
- Scikit-learn version is: 1.3.1

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
because through the project you can download from *Google Drive* a directory named *models*, where there are the saves of the pre-trained models and kfold result.

## Who ate my Chihuahua?

Artificial Neural Networks (ANNs) constitute a fundamental paradigm within the realm of artificial intelligence, inspired by the intricate structure and functioning of the human brain. Comprising interconnected neurons, ANNs excel in learning complex patterns and relationships from data, offering unparalleled versatility across various domains.

In this context, ANNs are adopted in the context of image classification, a fundamental task within computer vision, that involves assigning predefined labels or categories to input images.

This project explores binary image classification, a subset of image classification, focusing on the Muffins vs. Chihuahua dataset. The implementation involves leveraging Multilayer perceptron (MLP), Convolutional Neural Network (CNN), MobileNet, and VGG16 models within the Keras framework.

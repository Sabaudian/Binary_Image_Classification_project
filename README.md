# Statistical Methods for Machine Learning: experimental project
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white">

Use Keras to train a neural network for the binary classification of muffins and Chihuahuas.

## General Information

- **Python** version is: **3.10.5**
- **Scikit-learn** version is: **1.3.1**
- **Tensorflow** version is: **2.15.0**
- **requirements.txt** contains all the necessary python packages (_pip install -r requirements.txt_).
- The **models folder** contains the saves of the hyperparameters tuning and k-fold cross-validation processes.

## Structure of the project
the architecture of this project is fundamentally organized into four blocks. The initial two blocks are dedicated to preprocessing and data preparation tasks, whereas the latter two blocks are focused on model construction: classification and evaluation.

<p align="center">
  <img src="https://github.com/Sabaudian/SMML_project/assets/32509505/4b168037-0c91-4363-bcd5-cc720ae99e86">
</p>

#### A. Preprocessing
In the preprocessing phase, the emphasis is on refining the dataset. The process involves systematically addressing corrupted files, detecting and managing duplicates through image hashing, and conducting a thorough dataset check.

#### C. Classification
In the classification phase, a robust image classification pipeline is established using Keras and TensorFlow. The implementation introduces configurable models, including Multilayer Perceptron, Convolutional Neural Network and MobileNet. The workflow seamlessly integrates hyperparameter tuning and K-fold cross-validation for comprehensive model optimization.

#### D. Evaluation
In the evaluation phase, the model’s performance is systematically assessed through the presentation of insightful metrics, such as loss and accuracy. The module further generates detailed classification reports, produces confusion matrices, and offers intuitive plots for model predictions.

## Performace Summary:

|   | MLP | CNN | MOBILENET | 
| - | --- | ------------- | ------------------- |
| Accuracy (%)  | 73.057 | 92.821 | 99.155 |
| Loss  | 0.752 | 0.292 | 0.022 |
| F1-Score | 0.738 | 0.928 | 0.992 |


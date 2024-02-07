# Statistical Methods for Machine Learning: experimental project
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white">

Use Keras to train a neural network for the binary classification of muffins and Chihuahuas.

## General Information

- **Python** version is: **3.10.5**
- **Scikit-learn** version is: **1.3.1**
- **Tensorflow** version is: **2.15.0**
- **requirements.txt** contains all the becessary python packages.
- The **models folder** containes the saves of the hyperparameters tuning process.
  > If you want to performe the tuning, you can simply delete the folder.


## Who ate my Chihuahua?

The project primarily utilizes the **Keras** and **Tensorflow** frameworks to set up the work environment, construct, and train neural network models. Starting with crucial preprocessing steps to assess the quality of the data, identifying corrupted files, following by a check for the possible existence of duplicate files—both factors impacting the performance of the training models. Next to this quality assessment, the data undergo a preparation step, involving transformations such as adjusting size and color format of the images. Additionally, processes like data augmentation are applied to enhance results and overall performance.

This preprocessing step sets the stage for the subsequent exploration of various network architectures. To address the task at hand, we experiment with three distinct network architectures: The **Multilayer Perceptron (MLP)**, **Convolutional Neural Network (CNN)** and **MobileNet** model. 

Beyond architectural variations, our study incorporates **hyperparameter tuning** to enhance overall performance. The resulting optimal parameters and weights are saved and utilized as the basis for **K-Fold Cross-Validation** to calculate risk estimates, employing the **zero-one loss** function. At the end, the models are evaluated on the test set to have a final understanding of their generalization capabilities.

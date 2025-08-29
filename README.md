# **ResNet-50 Implementation in PyTorch for Image Classification**

This repository contains a PyTorch implementation of the ResNet-50 architecture for image classification. The model is trained and evaluated on the SIGNS dataset, which consists of images of hand gestures representing numbers from 0 to 5\.

## **Project Overview**

This project demonstrates the implementation of a deep residual network (ResNet-50) from scratch using PyTorch. It covers the essential steps of a deep learning project, including:

* Loading and preprocessing image data.  
* Defining the ResNet-50 architecture with its core building blocks (Identity and Convolutional blocks).  
* Training the model using the Adam optimizer and Cross-Entropy Loss.  
* Evaluating the model's performance on a validation and test set using accuracy and F1-score.

## **Table of Contents**

* Model Architecture  
* Dataset 
* Requirements
* Usage
* Results

## **Model Architecture**

The ResNet-50 model is built using two main types of residual blocks:

1. **Identity Block**: Used when the input and output dimensions are the same. It features a shortcut connection that skips over three convolutional layers.  
2. **Convolutional Block**: Used when the input and output dimensions differ. It includes a convolutional layer in the shortcut path to resize the input and match the output dimensions, allowing for feature map down-sampling.

The overall architecture is composed of five stages of convolutional and identity blocks, followed by an average pooling layer and a fully connected layer for classification.

## **Dataset**

The model is trained on the **SIGNS dataset**, which is loaded using the load\_dataset() function. The dataset contains images of size 64x64 pixels with 3 color channels (RGB). The task is to classify the hand gesture in each image into one of six classes (0-5).

The data is split into:

* **Training set**: Used to train the model.  
* **Validation set**: A subset of the training data used for hyperparameter tuning and monitoring performance during training.  
* **Test set**: Used for final evaluation of the trained model.

## **Requirements**

The following libraries are required to run the notebook:

* PyTorch  
* NumPy  
* Matplotlib  
* h5py  
* Pillow (PIL)  
* scikit-learn  
* SciPy

You can install the necessary packages using pip:

pip install torch numpy matplotlib h5py Pillow scikit-learn scipy

## **Usage**

1. **Clone the repository:**  
   git clone \<repository-url\>

2. **Navigate to the project directory:**  
   cd \<project-directory\>

3. **Ensure you have the dataset** and the resnets\_utils.py file in the same directory.  
4. **Run the Jupyter Notebook:**  
   jupyter notebook Resnet.ipynb

5. Execute the cells in the notebook sequentially to load the data, build the model, train it, and evaluate its performance.

## **Results**

The model was trained for 20 epochs. The performance on the validation set at the end of each epoch is shown below:

* ...  
* epoch 18: val acc \= 0.898, val F1 \= 0.897  
* epoch 19: val acc \= 0.917, val F1 \= 0.917  
* epoch 20: val acc \= 0.907, val F1 \= 0.905

After training, the model achieved the following performance on the test set:

* **Test Accuracy:** 94.2%  
* **Test F1-Score:** 0.940

A visual inspection of training and test samples is also provided in the notebook to qualitatively assess the dataset.

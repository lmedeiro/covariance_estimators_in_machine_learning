# Analyzing Estimators for Deep CNN Models

## Abstract
The goal of this work was to evaluate the performance of three different covariance matrix estimators in the context of Principal Component Analysis (PCA), Convolutional Neural Networks (CNNs), and classification. Namely, the different estimators were used to compute PCA features from images in the Fashion-MNIST dataset. These features were then fed to a simple CNN for a classification exercise. The different estimators, Sample Covariance, Ledoit-Wolf Linear Shrinkage, and Rotationally Invariant estimators are evaluated on the basis of the final classification performance.  


## Repo Description

[images folder](./images) 
    Contains images that are deemed sufficiently important to store. 

[logs folder](./logs)
    Contains logs related to training and validation

[trained models folder](./models)
    Contains models that have been trained

[Pre-processing resources](./PipelineResources.py)
    Contains functions that are used in the pre-processing

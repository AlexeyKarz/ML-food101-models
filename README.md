# Computer Vision - Food101 models    

**This repository contains models that were built using food101 dataset. The original source of data is [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).**


*Food Vision projects were made during compleeting the guides of [ZTM course](https://github.com/mrdbourke/tensorflow-deep-learning/tree/main). The models used, their architecture, function implementation, chosen classes etc. might differ.*


## The projects:

1. Binary food vision ([`binary_food_vision.ipynp`]()) - binary image classification: deep learning model which is built using Keras Sequential API. 

2. Fast Food Vision ([`fast_food_vision.ipynb`]()) - multiclass (10 classes) image classification: CNN (Convolutional Neural Network) deep learning model which is built using Keras Functional API.

3. Fast Food Vision 2.0 ([`transef_learning_fast_food_vision.ipynb`]()) - multiclass (10 classes) image classification: the model with the plroblem similar to the previous one, but now it is built using transef learning with further rine-tuning. Tested models are: . The best scored model - .



## Extra:
1. [`food_helper_functions.py`]() - the file with some helper functions (mostly for visualisation) that I use in other notebooks. Functions are based on ZTM course.
2. [`data_extracting.ipynb`]() - the file with functions I used to create different subsets of original food101 dataset. The functions are modified versions of [ZTM course file](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb).

##
*The Food-101 data set consists of images from Foodspotting which are not property of the Federal Institute of Technology Zurich (ETHZ). Any use beyond scientific fair use must be negociated with the respective picture owners according to the Foodspotting terms of use.*


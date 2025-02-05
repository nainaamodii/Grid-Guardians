"""
Digit Recognition model using mnist dataset
Building a neural network using tensor flow 
"""
import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

# loading the mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize pixel intensity in range 0-1
''' values are scaled down by dividing each value by max possible value,
ie pixel intensity 255 in case of images with 8-bit depth
Neural networks work better when inputs are scaled down,
this helps in faster convergence during training, 
and reduces computation time'''
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


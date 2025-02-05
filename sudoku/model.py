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

"""
# build the model 
model = tf.keras.models.Sequential()

# add layers in the model 
''' flatten 28x28 grid into one line of 784 elements'''
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) 

''' dense layer is most basic where every neuron is connected to each other'''
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'relu'))

''' softmax ensures that all outputs add to 1, gives probability'''
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# compile the model
''' loss function is categorical cross entropy, optimizer is adam, metrics is accuracy '''
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# fit data into model
'''epoch is how many times data will be seen again - iterations'''
model.fit(x_train, y_train, epochs = 3)

# save the model
model.save('digit_recognition.keras')

"""

# load the saved model 
model = tf.keras.models.load_model('digit_recognition.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

# img = cv2.imread("digits/digit1.png")[:,:,0]
# img = np.invert(np.array([img]))
# prediction = model.predict(img)
# print(f"prediction: {np.argmax(prediction)}")
# plt.imshow(img[0], cmap=plt.cm.binary)
# plt.show()

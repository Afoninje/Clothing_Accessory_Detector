import tensorflow as tf
from tensorflow import keras
import numpy as numpy
import matplotlib.pyplot as plt

#Get data from the fashion_mnist library
data = keras.datasets.fashion_mnist

#Split the data into train and test data
#The images are 28*28 pixels , with each pixel having the value between 0-255
#Value 0 = White & Value 255 = Black
#The labels are numbers betweeen 0-9 , each label corresponds to a type of clothing
(train_images, train_labels),(test_images,test_labels) = data.load_data()

#The labels in the fashion_mnist are between 0-9 , this gives a name to them
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Shrink the RGB value of each pixel from 0-255 to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

#Create the neural network

model = keras.Sequential([
    #Input layer: Takes 28*28 inputs i.e each pixel in the image
    keras.layers.Flatten(input_shape=(28,28)),
    #Hidden layer 1: Will help calculate the output
    keras.layers.Dense(128, activation="relu"),
    #Output layer: Will highlight the node to which the image corresponds to
    #Softmax ensures the addition of all the nodes is 1
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train the neural network
# Epochs = Number of times each images will be shown to the network
#verbose = disable the command to keep printing values
model.fit(train_images , train_labels , epochs=5,verbose=0)

#Get the output

loss , accuracy = model.evaluate(test_images,test_labels, batch_size=10000,verbose=0)
print(accuracy)




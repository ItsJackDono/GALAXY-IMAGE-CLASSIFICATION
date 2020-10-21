# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:21:47 2020

@author: itsmr
"""

#importating the neccassary librarys needed for creating a CNN model, ploting data, saving history and testing data
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import csv
import warnings
from keras import backend as K
import os
import timeit
import matplotlib.pyplot as plt
import pandas as pd
#stop some of the warnings popping up which stops the flow of the console, esaier to read what is currently happening at each stage
warnings.filterwarnings('ignore')


with open('C:/Users/itsmr/Desktop/CS 3rd Year/training_solutions_rev1.csv', newline='') as csv_file:
    csv_reader = pd.read_csv(csv_file)
#creating a class array containig all the classes from the solutions file 
    classes = [
        'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1',
        'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3',
        'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',
        'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6',
        'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2',
        'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
        'Class11.5', 'Class11.6']
#This refererenced code allows for each row in the ID column of the solutions csv file to be used as the starting point of the image name by then,
#finishing the string with the .jpg, this function will find the corropsonding image to the same image ID with each of the 37 weighted probabilities
#(J, 2018) 
    def append_ext(fn):
        return fn + ".jpg"
#(J, 2018)
csv_reader["id"] = csv_reader['GalaxyID'].astype(str).apply(append_ext)

#specifying the image dimensions before passing through the input layer in the CNN model
IMG_WIDTH = 256
IMG_HEIGHT = 256
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)


#a timer to allow for recorded results
start = timeit.default_timer()
#the image data generator class was cruical for the data augmentation
#this use of ratiting , changing the shear and zoom range allowed for new images to be looked at

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    shear_range=0.2,
    zoom_range=0.2,
    #this validation split will change how the training and validation splits the images between both training stages
    #speicifcally 20% of the images will be used for vlidation as a way of testing the training data\
    validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1. / 255)
#both training sets use the image training folder but due the validation split 20% can be soley used for validation, like testing the training data
training_set = train_datagen.flow_from_dataframe(
    dataframe=csv_reader,
    directory='C:/Users/itsmr/Desktop/CS 3rd Year/images_training_rev1',
    x_col="id",
    #the classes array is set to the y column to represent each class
    y_col=classes,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    #here the class array is used to define the classes used by class mode input is raw for the data as there is no definitive class
    classes=classes,
    color_mode='grayscale',
    batch_size=60,
    shuffle=True,
    class_mode='raw',
    subset='training')
#setting the valditation set 
valid_set = train_datagen.flow_from_dataframe(
    dataframe=csv_reader,
    directory='C:/Users/itsmr/Desktop/CS 3rd Year/images_training_rev1',
    x_col="id",
    #the classes array is set to the y column to represent each class
    y_col=classes,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    #same as for the training set, the class array is used to define the classes used by class mode input is raw for the data as there is no, 
    #definitive class
    classes=classes,
    color_mode='grayscale',
    batch_size=60,
    shuffle=True,
    class_mode='raw',
    subset='validation')



#following the design scheme the use of keras allows for the creating and following the design easier
#higher amounts where chosen for conv2D layers but due to hardware constraints this was limited
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(37, activation='sigmoid'))
model.summary()

#this can be changed but the default value for adam starts at 0.001
#Better for testing
learningRate = 0.001
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learningRate),
              metrics=['accuracy'])



# Train the model
#various epochs where chosen but the best found was 10 as after a few iterations no dramatic difference in training and validation occured
#the demanding toll on the CPU pushed the rest of the parameters to be set in a way The pc hardware could just about handle
classifier = model.fit_generator(
    training_set,
    steps_per_epoch=10,
    epochs=10,
    validation_data=valid_set,
    validation_steps=30
)


# Visualize training results
acc = classifier.history['accuracy']
val_acc = classifier.history['val_accuracy']

loss = classifier.history['loss']
val_loss = classifier.history['val_loss']
#printing the total time of the model to see how long each iteration takes
print("Time Taken to run the model:", timeit.default_timer() - start)
#printing these keys helped plot the data onto the graphs for testing
print(classifier.history.keys())

epochs_range = range(10)
#plotting the training accuracy and validation accuracy onto a graph based on iterations of epochs
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

#plotting the training loss and validation loss onto a graph based on iterations of epochs
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


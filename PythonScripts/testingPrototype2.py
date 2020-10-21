    
# coding: utf-8
#importating the neccassary librarys needed for creating a CNN model, ploting data, saving history and testing data
import keras
#new library for  storing the weights after the model has been trained
import h5py
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
warnings.filterwarnings('ignore')



IMG_WIDTH = 256
IMG_HEIGHT = 256
input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)

#following the design scheme the use of keras allows for the creating and following the design easier
#higher amounts where chosen for conv2D layers but due to hardware constraints this was limited further with CPU reaching 90% usage 
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(37, activation='sigmoid'))
model.summary()



#loading teh h5py weights, unable to know if these are properly being used
model.load_weights("C:/Users/itsmr/Desktop/CS 3rd Year/models.h5")

	
#batch file can be changed here to test the limits of the hardware as well
batch_size=64
# the gray scale image width at which prediction occurs
IMG_SHAPE=(256, 256)
# fucnstion which reads the input image and return grayscale
def read_image(path, shape):
    x = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, shape)
	#normalising
    x = x/255.
    return x.reshape((256, 256,1))

def test_image_generator(ids, shape=IMG_SHAPE):
    x_batch = []
    for i in ids:
        x = read_image('C:/Users/itsmr/Desktop/CS 3rd Year/images_test_rev1/'+i, shape=IMG_SHAPE)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch
# all the test images are loaded in to list
val_files = os.listdir('C:/Users/itsmr/Desktop/CS 3rd Year/images_test_rev1/')
val_predictions = []
#number of validation files
N_val = len(val_files)
# the images are passed as batches according to the batch size given
for i in tqdm(np.arange(0, N_val, batch_size)):
    if i+batch_size > N_val:
        upper = N_val
    else:
        upper = i+batch_size
    X = test_image_generator(val_files[i:upper])
	# predicting the batches
    y_pred = model.predict(X)
    val_predictions.append(y_pred)


#this reads the empty benchmark csv file already containing the rows and collumns to write new test predictions which 
#includes the column names and classes
df=pd.read_csv('C:/Users/itsmr/Desktop/CS 3rd Year/all_zeros_benchmark.csv')
val_predictions = np.array(val_predictions)

#the predictions need be stacked vertically here
Y_pred = np.vstack(val_predictions)
#getting the list of images ids using a different method previously made in ana ealier prototype as writing to an empty file is needed
#these image ids will be reshaped into a hroizontal array and stored in the empy csv file
ids = np.array([v.split('.')[0] for v in val_files]).reshape(len(val_files),1)
final_df = pd.DataFrame(np.hstack((ids, Y_pred)), columns=df.columns)
# the prediction for each id with their class value is written into csv file
final_df.to_csv('all_zeros_benchmark.csv')


# this function was created last for testing an image to a class id probabiltiies
#uses a similar function as only reading this file is needed
def read_image_by_id(id):	
	final_df=pd.read_csv('all_zeros_benchmark.csv')
	result=final_df[final_df['GalaxyID']==str(id)]
	img=plt.imread('C:/Users/itsmr/Desktop/CS 3rd Year/images_test_rev1/'+str(id)+'.jpg')
	plt.imshow(img)
	plt.show()
	return result
#tesing predication credability with the image, this will be sent 
read_image_by_id(100018)
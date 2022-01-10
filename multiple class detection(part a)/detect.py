import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(222, activation='softmax'))


model.load_weights('model.h5')

    
im=cv2.imread("6.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)   
prediction = model.predict(cropped_img)
maxindex = int(np.argmax(prediction))
maxindex1 = int(np.argmax(prediction-prediction[maxindex]))
maxindex2 = int(np.argmax(prediction-prediction[maxindex1]))
maxindex3 = int(np.argmax(prediction-prediction[maxindex2]))
maxindex4 = int(np.argmax(prediction-prediction[maxindex3]))
print(prediction)

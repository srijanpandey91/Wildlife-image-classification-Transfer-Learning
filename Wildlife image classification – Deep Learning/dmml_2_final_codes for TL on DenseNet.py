# -*- coding: utf-8 -*-
"""dmml 2 final codes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZC4rN69KTfiRryu-CWLQXZlmyn1FChE8

WILDLIFE IMAGE CLASSIFICATION USING DEEP LEARNING TECHNIQUES

Image Classification using Densenet
"""

a = []
while(1):
    a.append('1')

"""Code for connecting Colab to Google Drive"""

#!pip install -U -q PyDrive
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials

#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

#from google.colab import drive
#drive.mount('/wildlife')

"""Importing all the required libraries"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from keras.utils import plot_model
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import tensorflow as tf

"""**Exploratory data Analysis**"""

# for getting number of classes
folders = glob('/content/drive/My Drive/wildlife/Training/*')
folders

from skimage.io import imread
img = imread('/content/drive/My Drive/wildlife/Training/deer/0086a3b72f62599ad6.jpg')

plt.imshow(img)
plt.axis('off')
plt.show()

from IPython.display import Image, display
Image('/content/drive/My Drive/wildlife/Training/bald_eagle/00e148aeea989ba56b.JPG')

data_set = '/content/drive/My Drive/wildlife'

labels = os.listdir(data_set)
print("Number of Labels:", len(labels))

total = 0
for lb in os.scandir(data_set):
    print('folder: {} images: {}'.format(lb.name, len(os.listdir(lb))))
    total += len(os.listdir(lb))
print('Total images:', total)

data_set = '/content/drive/My Drive/wildlife/Training'

labels = os.listdir(data_set)
print("Number of Labels:", len(labels))

total = 0
for lb in os.scandir(data_set):
    print('folder: {} images: {}'.format(lb.name, len(os.listdir(lb))))
    total += len(os.listdir(lb))
print('Total images:', total)

data_set = '/content/drive/My Drive/wildlife/Test'

labels = os.listdir(data_set)
print("Number of Labels:", len(labels))

total = 0
for lb in os.scandir(data_set):
    print('folder: {} images: {}'.format(lb.name, len(os.listdir(lb))))
    total += len(os.listdir(lb))
print('Total images:', total)

"""**Initialisation of Densenet**"""

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/content/drive/My Drive/wildlife/Training'
valid_path = '/content/drive/My Drive/wildlife/Test'

# add preprocessing layer to the front of VGG
den = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in den.layers:
  layer.trainable = False

# our layer
x = Flatten()(den.output)
prediction = Dense(len(folders), activation='softmax',  kernel_regularizer=regularizers.l2(0.0001))(x)  #using L2 regularizer to avoid overfitting.

# create a model object
model = Model(inputs=den.input, outputs=prediction)

# view the structure of the model
model.summary()

# compiling the model with cost and optimization method
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Data Augmentation before fitting our model
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/My Drive/wildlife/Training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/drive/My Drive/wildlife/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fitting the model
result = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))

# loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.title('model loss with densenet')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.title('model accuracy with densenet')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#Saving our model
model.save('densenet_model.h5')

## Predicitng Models
model = load_model('densenet_model.h5')
img = image.load_img('/content/drive/My Drive/wildlife/Test/deer/cac897d1b08f1fc615.JPG', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

classes

"""The image used was of a deer which was our 5th class. The model predict the fifth class as 1."""

#Plotting our model
plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB', expand_nested=True)
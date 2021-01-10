#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import matplotlib.pyplot as plt


# In[2]:


classifier = Sequential()


# In[3]:


classifier.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))


# In[4]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[5]:


classifier.add(Flatten())


# In[6]:


classifier.add(Dropout(0.2))


# In[7]:


classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 5, activation = 'softmax'))


# In[8]:


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# In[9]:


classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[10]:


from keras.preprocessing.image import ImageDataGenerator


# In[11]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'D:\DMML datasets/wildlife/Training',
        target_size=(150, 150),
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'D:\DMML datasets/wildlife/Test',
        target_size=(150, 150),
        batch_size=64,
        class_mode='categorical')


# In[12]:


history = classifier.fit_generator(
        train_generator,
        steps_per_epoch=2905,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=724)


# In[13]:


print(history.history.keys())


# In[14]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()


# In[15]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()


# In[ ]:





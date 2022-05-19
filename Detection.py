#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow import keras
from keras import layers
from keras.models import Sequential


# In[15]:


batch_size = 1000
img_height = 200
img_width = 200
data_dir = "C:\\Users\\saime\\OneDrive\\Desktop\\TrainingData\\"


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.15,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size = batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.15,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size= batch_size)

class_names = train_ds.class_names
print(class_names)


# In[16]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# In[17]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# In[18]:


epochs=75
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[7]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[8]:


result = model.predict(train_ds)
print(result)


# In[9]:


img = tf.keras.utils.load_img(
    "C:\\Users\\saime\\Downloads\\cat1.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[10]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
imgplot = plt.imshow(img)


# In[11]:


model.compile(optimizer="adam", loss="mean_squared_error")


# In[12]:


print(img_array.shape)
print(predictions)
print(score)


# In[13]:


import cv2


# In[14]:


def Predictor(imga):
    #img = tf.keras.utils.load_img(
    #"C:\\Users\\saime\\Downloads\\cat1.jpg", target_size=(img_height, img_width))
    
    #img = tf.keras.utils.load_img(imga,target_size=(img_height,img_width))
    dim = (img_height,img_width)
    resized = cv2.resize(imga, dim, interpolation = cv2.INTER_AREA)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

#print(
#    "This image most likely belongs to {} with a {:.2f} percent confidence."
#    .format(class_names[np.argmax(score)], 100 * np.max(score))
#)
    trial = class_names[np.argmax(score)]
    return trial


# In[15]:


tf.reshape(img_array, [200, 200,3])


# In[16]:


import winsound
import pyttsx3


# In[17]:


cap = cv2.VideoCapture(0)
#loaded_model = load_model("/content/Smart_Alarm_System.h5")



while True:
  suc,img = cap.read()

  img1 = tf.expand_dims(tf.image.resize(img, (200, 200)), 0)
  score = model.predict(img1)
  score = tf.nn.softmax(score)
  print(np.argmax(score))
#   print(img.shape)
  print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
  
  cv2.putText(img,(class_names[np.argmax(score)]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,2,255)



    
  cv2.imshow("Image",img)
  
  cv2.waitKey(1)


video.release()
cv2.destroyAllWindows()


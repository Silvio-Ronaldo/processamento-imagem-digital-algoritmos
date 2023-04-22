import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras import layers,models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras import optimizers, losses

inputData = "./AugmentedAlzheimerDataset"

image_dir = Path(inputData)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) 

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)


# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()


# Train test split
train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split

train_images = train_datagen.flow_from_directory(
    inputData,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data

validation_images = train_datagen.flow_from_directory(
    inputData , # same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # set as validation data

vgg_model = Sequential()

vgg = keras.applications.VGG16(input_shape=(224,224,3), include_top = False, weights= 'imagenet')
vgg_model.add(vgg)

x = vgg_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(3078,activation='relu')(x) 
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(256,activation='relu')(x) 
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(4,activation='softmax')(x)
tf_model = keras.models.Model(inputs=vgg_model.input,outputs=out)

for layer in vgg.layers:
  layer.trainable=True

tf_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
history_vgg = tf_model.fit(train_images, 
                    steps_per_epoch=len(train_images), 
                    epochs = 10, 
                    validation_data = validation_images, 
                    validation_steps=len(validation_images))

#Accuracy
plt.plot(history_vgg.history['accuracy'])
plt.plot(history_vgg.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# loss
plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
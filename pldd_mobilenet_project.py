# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import keras
from keras import backend as K
# from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image

#from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
# from IPython.display import Image




import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.enable_eager_execution()

import tensorflow_hub as hub
import os
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from keras import optimizers

import os
img_path = os.getcwd() + r'/images'
print(img_path)

base_dir = os.getcwd()+r'/content/drive/MyDrive/RLD_updated'
train_dir = os.path.join('/content/drive/MyDrive/RLD_updated/Train')
validation_dir = os.path.join('/content/drive/MyDrive/RLD_updated/Val')
test_dir = os.path.join('/content/drive/MyDrive/RLD_updated/test')

train_Blast_dir = os.path.join(train_dir, 'Blast')
train_Blight_dir = os.path.join(train_dir, 'Blight')
train_Brownspot_dir = os.path.join(train_dir, 'Brownspot')
train_Tungro_dir = os.path.join(train_dir, 'Tungro')


validation_Blast_dir = os.path.join(validation_dir, 'Blast')
validation_Blight_dir = os.path.join(validation_dir, 'Blight')
validation_Brownspot_dir = os.path.join(validation_dir, 'Brownspot')
validation_Tungro_dir = os.path.join(validation_dir, 'Tungro')







NUMBER_OF_CLASS = 4

IMAGE_SIZE = (224, 224)
def prepare_image(file):
    img = image.load_img(file, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from keras.preprocessing import image
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

mobile = keras.applications.mobilenet.MobileNet()

#imports the mobilenet model and discards the last 1000 neuron layer.
base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(224,224,3))

x=base_model.output
x=GlobalAveragePooling2D()(x)
#add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 1
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
print(base_model.input)

for i,layer in enumerate(model.layers):
  print(i,layer.name)

for layer in model.layers[:88]:
    layer.trainable=False
for layer in model.layers[88:]:
    layer.trainable=True

model.summary()

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

BATCH_SIZE = 32

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30, ###40,
    width_shift_range=0.1, ###0.2,
    height_shift_range=0.1, ###0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the val data should not be augmented!
test_datagen= ImageDataGenerator    (rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=True)

# Flow validation images in batches using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

import scipy
EPOCHS = 10
step_size_train=train_generator.n//train_generator.batch_size
step_size_validation=validation_generator.n//validation_generator.batch_size
print(step_size_train)
print(step_size_validation)
history = model.fit(train_generator, steps_per_epoch=step_size_train, epochs=EPOCHS,
 validation_data=validation_generator, validation_steps=step_size_validation,
 shuffle=True)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of loss results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', label='training')
plt.plot(epochs, val_acc, 'b', label='validation')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', label='training')
plt.plot(epochs, val_loss, 'b', label='validation')
plt.title('Training and validation loss')
plt.legend()

os.mkdir ('/content/drive/MyDrive/output14/')

model.save("/content/drive/MyDrive/output14/savemodel")

#model = tf.keras.models.load_model('/content/drive/MyDrive/mobilenet_1_0_224_tf.h5')
#model.summary()



#model_path = '/content/drive/MyDrive/mobilenet_1_0_224_tf_no_top.h5'
#model = tf.keras.models.load_model(model_path)

# Use the loaded model for predictions or other tasks
#predictions = model.predict(input_data)

class_names=['Blast', 'Blight' , 'Brownspot' , 'Tungro']

# T0 print the Classification Report
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_generator.classes, y_pred))

"""to run the test code
the test data of different images should be in single folder
"""

import glob
list_of_files = glob.glob(test_dir+'/*.jpg')           # create the list of file from test directory
print(len(list_of_files))
print(test_dir)

# Randomly choose a picture to test your model
import matplotlib.pyplot as plt

img_number= int(np.random.randint(0,len(list_of_files), size=1))
test_image=list_of_files[img_number]

new_image = load_img(test_image)
new_image = new_image.resize((224, 224))

img_array = img_to_array(new_image)
img_batch = np.expand_dims(img_array, axis=0)
pred = model.predict(img_batch)
print(pred[0])
prediction_result = class_names[np.argmax(pred[0])]
print('prediction=',prediction_result)



plt.imshow(new_image)

# To Evaluate the model with test images
score = model.evaluate(train_generator, batch_size=1, verbose=1)

# Graph between Training Accuracy and Validation Accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(history.epoch, acc, 'r', label='Training accuracy')
plt.plot(history.epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.figure()

# Graph between Training Loss and Validation Loss
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(history.epoch, loss, 'r', label='Training Loss')
plt.plot(history.epoch, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

# Graph for Learning rate ~ Exponential Decay
plt.plot(history.epoch, history.history["lr"], "o-")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(" exponential_decay", fontsize=14)
plt.grid(True)
plt.show()

# T0 print the Classification Report
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_generator.classes, y_pred))

# To print the Confusion Matrix
cf_matrix = confusion_matrix(test_generator.classes, y_pred)
Labels = ['Brown spot', 'Leaf smut', 'Bacterial leaf blight']
plt.figure(figsize=(8, 8))
heatmap = sns.heatmap(cf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d', color='blue')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from google.colab.patches import cv2_imshow

with_mask_files = os.listdir('./images/0')
print(with_mask_files[0:5])
print(with_mask_files[-5:])

without_mask_files = os.listdir('./images/1')
print(without_mask_files[0:5])
print(without_mask_files[-5:])

print('Number of with mask images:', len(with_mask_files))
print('Number of without mask images:', len(without_mask_files))

with_mask_labels = [1]*2083
without_mask_labels = [0]*2231
print(with_mask_labels[0:5],len(with_mask_labels))
print(without_mask_labels[0:5],len(without_mask_labels))
labels = with_mask_labels + without_mask_labels
print(len(labels))

img = mpimg.imread('./images/0/ (101).png')
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('./images/1/ (1).png')
imgplot = plt.imshow(img)
plt.show()

with_mask_path = './images/0/'
data = []
for img_file in with_mask_files:
  image = Image.open(with_mask_path + img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)
without_mask_path = './images/1/'
for img_file in without_mask_files:
  image = Image.open(without_mask_path + img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)
X = np.array(data)
Y = np.array(labels)
type(X),type(Y)
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X.shape, X_train.shape, X_test.shape)

X_train_scaled = X_train/255
X_test_scaled = X_test/255
num_of_classes = 2
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)

h = history
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()

input_image_path = './images/msk.jpg'
input_image = cv2.imread(input_image_path)
cv2_imshow(input_image)
input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized/255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)
if input_pred_label == 1:
   print('The person in the image is wearing a mask')
else:
   print('The person in the image is not wearing a mask')
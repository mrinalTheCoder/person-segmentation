
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Lambda, MaxPool2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras.callbacks import LearningRateScheduler
from IPython.display import Image, display
import cv2
import numpy as np
import os
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# getting permission to get files from Google Drive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# download and unzip images and masks
downloaded = drive.CreateFile({'id':"1NU-skxQaoE7CphcS_KfsuiqSXJG1sC1U"})
downloaded.GetContentFile("img.zip")
with zipfile.ZipFile("img.zip", 'r') as zip_ref:
    zip_ref.extractall("img")

downloaded = drive.CreateFile({'id':"1O1JeORPoGmVdRf8G_L9MME__rbp4xBph"})
downloaded.GetContentFile("masks.zip")
with zipfile.ZipFile("masks.zip", 'r') as zip_ref:
    zip_ref.extractall("masks")

downloaded = drive.CreateFile({'id':"149nbJpCK6Ezlg7HMxEIT-vJOwV-9iD0U"})
downloaded.GetContentFile("masks_val.zip")
with zipfile.ZipFile("masks_val.zip", 'r') as zip_ref:
    zip_ref.extractall("masks_val")

downloaded = drive.CreateFile({'id':"1XZDIzBk_NqZk4ONK_rsL8mFFCyv_3BUX"})
downloaded.GetContentFile("img_val.zip")
with zipfile.ZipFile("img_val.zip", 'r') as zip_ref:
    zip_ref.extractall("img_val")


num_images = len(os.listdir("img/img")) #number of training images
img_size = 224
x_train = np.array([[[[]]]])
y_train = np.array([[[[]]]])
# we read every image, resize it and add it to a numpy array.
for i in os.listdir("img/img"):
    img = cv2.imread("img/img/" + i)
    x_train = np.append(x_train, cv2.resize(img, (img_size, img_size)))

# masks are also read and resized
for i in os.listdir("masks/masks"):
    img = cv2.imread("masks/masks/" + i, 0)
    y_train = np.append(y_train, cv2.resize(img, (img_size, img_size)))

# the masks have grayscale value 0 or 2. We need 0 or 1
for i in range(len(y_train)):
    if y_train[i] >= 1:
        y_train[i] = 1

# reshaping after appending data
x_train = np.reshape(x_train, (num_images, img_size, img_size, 3))
y_train = np.reshape(y_train, (num_images, img_size, img_size))

num_images = len(os.listdir("img_val/img_val"))
img_size = 224
x_val = np.array([[[[]]]])
y_val = np.array([[[[]]]])
for i in os.listdir("img_val/img_val"):
    img = cv2.imread("img_val/img_val/" + i)
    x_val = np.append(x_val, cv2.resize(img, (img_size, img_size)))

for i in os.listdir("masks_val/masks_val"):
    img = cv2.imread("masks_val/masks_val/" + i, 0)
    y_val = np.append(y_val, cv2.resize(img, (img_size, img_size)))

for i in range(len(y_val)):
    if y_val[i] >= 1:
        y_val[i] = 1

x_val = np.reshape(x_val, (num_images, img_size, img_size, 3))
y_val = np.reshape(y_val, (num_images, img_size, img_size))

# U-net architecture
inputs = Input((img_size, img_size, 3))
s = Lambda(lambda x: x / 255) (inputs) #normalise values between 0 and 1

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPool2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPool2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPool2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPool2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
model.summary()

# training the model
model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=30, batch_size=4)

# which img to use for testing. should be in the range 0-58
im_number = 15
#show the original image
img = x_val[im_number]
cv2.imwrite("image_15.png", img)
display(Image("image_15.png"))

#display ground truth
ground_truth = y_val[im_number]*255
cv2.imwrite("ground_truth_15.png", ground_truth)
display(Image("ground_truth_15.png"))

# adding dimension as model requires batched data
img_to_model = np.expand_dims(img, axis=0)

# getting model predictions and removing axes for displaying
preds = model.predict(img_to_model)
preds = np.reshape(preds, (img_size, img_size))
# model returns float values. We are chainging them to 0 or 1
for i in range(len(preds)):
    for j in range(len(preds[i])):
        if preds[i][j] >= 0.5:
            preds[i][j] = 1
        else:
            preds[i][j] = 0

cv2.imwrite("preds_15.png", preds*255)
display(Image("preds_15.png"))

preds_3channel = np.stack((preds,)*3, axis=-1)
out_img = np.where(preds_3channel==0, preds_3channel, img)

cv2.imwrite("final_out_15.png", out_img)
display(Image("final_out_15.png"))

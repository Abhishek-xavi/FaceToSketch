import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from helper import *

from image_similarity_measures.quality_metrics import rmse, ssim
import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model

#Size of the image
SIZE = 256

cv_img = import_images("./CUHK Dataset/CUHK_training_cropped_photos/*.jpg", SIZE)
cv_sktch = import_images("./CUHK Dataset/CUHK_training_cropped_sketches/*.jpg", SIZE)
cv_img_test = import_images_test("./CUHK Dataset/CUHK_testing_cropped_photos/*.jpg")
cv_sktch_test = import_images_test("./CUHK Dataset/CUHK_testing_cropped_sketches/*.jpg")


#Splitting test and train
train_sketch_image = cv_sktch
train_image = cv_img
test_sketch_image = cv_sktch_test
test_image = cv_img_test

#Creating an np array of all the images.
train_sketch_image = np.reshape(train_sketch_image,(len(train_sketch_image),SIZE,SIZE,3))
train_image = np.reshape(train_image, (len(train_image),SIZE,SIZE,3))
print('Train color image shape:',train_image.shape)

test_sketch_image = np.reshape(test_sketch_image,(len(test_sketch_image),SIZE,SIZE,3))
test_image = np.reshape(test_image, (len(test_image),SIZE,SIZE,3))
print('Test color image shape',test_image.shape)

#Defining a function for downsample.
#This consists of a convolutional layer
#A batch normalization if flag is set to true
#A LeakyRelu Activation

def downsample(filters, size, stride = 2, apply_batch_normalization = True):
    #Creating a sequential model object
    downsample = tf.keras.models.Sequential()
    #Adding a convolutional layer to the model
    downsample.add(keras.layers.Conv2D(filters = filters, kernel_size = size, strides = stride, use_bias = False, kernel_initializer = 'he_normal', padding = 'same'))
    #Checking if batch normalisation is required.
    if apply_batch_normalization:
        downsample.add(keras.layers.BatchNormalization())
    #Adding activation function = LeakyReLU
    downsample.add(keras.layers.LeakyReLU())
    return downsample

#Defining a function for upsample.
#This consists of a convolutional layer
#A dropout layer if flag is set to true
#A LeakyRelu Activation

def upsample(filters, size, stride = 2, apply_dropout = False):
    #Creating a sequential model object
    upsample = tf.keras.models.Sequential()
    #Adding a transpose convolutional layer
    upsample.add(keras.layers.Conv2DTranspose(filters = filters, kernel_size = size, strides = stride, use_bias = False, kernel_initializer = 'he_normal', padding = 'same'))
    #Adding a dropout.
    if apply_dropout:
        upsample.add(tf.keras.layers.Dropout(0.01))
    #Adding a activation function
    upsample.add(tf.keras.layers.LeakyReLU()) 
    return upsample

# The encoding process
input_img = Input(shape=(256, 256, 3))  

############
# Encoding #
############

x = downsample(64, 3, apply_batch_normalization=False)(input_img)
x = downsample(64, 3, MaxPooling=True)(x)
x = downsample(128, 3)(x)
x = downsample(128, 3, MaxPooling=True)(x)
x = downsample(256, 3, apply_batch_normalization=False)(x)
x = downsample(256, 3)(x)
x = downsample(256, 3, MaxPooling=True)(x)
x = downsample(512, 3, apply_batch_normalization=False)(x)
x = downsample(512, 3)(x)
x = downsample(512, 3, MaxPooling=True)(x)
x = downsample(512, 3, apply_batch_normalization=False)(x)
x = downsample(512, 3)(x)
encoded = downsample(512, 3, MaxPooling=True)(x)


############
# Decoding #
############

decoder_input = upsample(512,3)(encoded)
x = upsample(256,3,False)(decoder_input)
x = upsample(128,3)(x)
x = upsample(64,3, False)(x)
x = upsample(32,3, False)(x)
decoded = tf.keras.layers.Conv2DTranspose(3,(2,2),strides = (1,1), activation='sigmoid', padding = 'same')(x)


# Declare the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss='mean_absolute_error', metrics=["accuracy"])

autoencoder.summary()


#Training the model

history = autoencoder.fit(train_image, train_sketch_image,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(test_image, test_sketch_image)
               )

#Displaying the generated images

ls = [i for i in range(0,95,8)]
for i in ls:
    predicted =np.clip(autoencoder.predict(test_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
    show_images(test_image[i],test_sketch_image[i],predicted)


#Calculating image similarity

rmse1 = []
ssim1 = []
preds = []
example_inputs = []
example_targets = []
for i in range(100):
  pred = np.clip(autoencoder.predict(test_image[i].reshape(1,SIZE,SIZE,3)),0.0,1.0).reshape(SIZE,SIZE,3)
  preds.append(pred)
  rmse1.append(rmse(np.asarray(test_image[i]),np.asarray(pred)))
  ssim1.append(ssim(np.asarray(test_image[i]),np.asarray(pred)))

#Getting rmse of images
print('rmse between images ', sum(rmse1)/len(rmse1))
#Getting ssim of images
print('SSIM between images ', sum(ssim1)/len(ssim1))

#PLotting model loss
plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.legend(loc="upper right")
plt.show()
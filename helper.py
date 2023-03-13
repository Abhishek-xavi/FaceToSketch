from matplotlib import pyplot as plt
import cv2 as cv
import glob
from tqdm import tqdm
from keras.utils import img_to_array
import tensorflow as tf


def show_images(real,sketch, predicted):
    plt.figure(figsize = (12,12))
    plt.subplot(1,3,1)
    plt.title("Image",fontsize = 15, color = 'Lime')
    plt.imshow(real)
    plt.subplot(1,3,2)
    plt.title("sketch",fontsize = 15, color = 'Blue')
    plt.imshow(sketch)
    plt.subplot(1,3,3)
    plt.title("Predicted",fontsize = 15, color = 'gold')
    plt.imshow(predicted)



def import_images(image_location,SIZE):
    
    #List to store images
    cv_img = []
    dataset_url =image_location
    path = glob.glob(dataset_url)
    path.sort()
    
    #Augmenting images
    for img in tqdm(path):
        image = cv.imread(img)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (SIZE,SIZE))
        image = image.astype('float32') /255.0
        cv_img.append(img_to_array(image))

        img1 = cv.flip(image, 1)
        cv_img.append(img_to_array(img1))

        img2 = cv.flip(image, -1)
        cv_img.append(img_to_array(img2))

        img3 = cv.flip(image, -1)
        img3 = cv.flip(img3, 1)
        cv_img.append(img_to_array(img3))

        img4 = cv.flip(image, -1)
        img4 = cv.flip(img4, -1)
        cv_img.append(img_to_array(img4))

        img5 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        cv_img.append(img_to_array(img5))

        img6 = cv.flip(img5, 1)
        cv_img.append(img_to_array(img6))

        img7 = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
        cv_img.append(img_to_array(img7))

        img8 = cv.flip(img7, 1)
        cv_img.append(img_to_array(img8))
    
    return cv_img


def import_images_test(image_location):
    
    #List to store images
    cv_img = []
    dataset_url =image_location
    path = glob.glob(dataset_url)
    path.sort()
    
    #Augmenting images
    for img in tqdm(path):
        image = cv.imread(img)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (256,256))
        image = image.astype('float32') /255.0
        cv_img.append(img_to_array(image))
    
    return cv_img


def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)

  return input_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image, IMG_HEIGHT=256, IMG_WIDTH=256):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image


def load_image_train(input_image, real_image):
  input_image, real_image = random_jitter(input_image, real_image)
  return input_image, real_image


def load_image_test(input_image, real_image,  IMG_HEIGHT=256, IMG_WIDTH=256):
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  return input_image, real_image


import numpy as np
import pickle, os
import cv2

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .. import config


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
width=256
height=256
depth=3

label_list = []

print("[INFO] Loading images ...")

input_path = os.path.join(config.dataset_path(), "PlantVillage")
plant_disease_folder_list = os.listdir(input_path)

print(plant_disease_folder_list)


for plant_disease_folder in plant_disease_folder_list:
    print(f"[INFO] Processing {plant_disease_folder} ...")

    plant_disease_image_list = os.listdir(os.path.join(input_path, 
                            plant_disease_folder))
        

    for image in plant_disease_image_list[:20]:
        image_directory = os.path.join(config.dataset_path(), "p", image)
        if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
            print(image_directory)
            #a = cv2.imread(image_directory)
            #print(a.shape())
            #print("image_shape: ", a.shape())
            label_list.append(plant_disease_folder)

print("[INFO] Image loading completed")  

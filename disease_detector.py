#import json
#import urllib.request
import numpy as np
#import pickle as pk
import tensorflow 
from keras.models import load_model
from keras.applications.vgg16 import VGG16
#from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array, load_img, array_to_img
#import keras.utils.data_utils
#import cv2
#import os
import json
#import h5py
import urllib.request
import numpy as np
import pickle as pk
import keras
from IPython.display import Image, display, clear_output
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from PIL import Image




model1 = VGG16(weights = 'imagenet')
model2 = load_model(r"C:\Users\Shreyas Desai\Documents\Plant-Disease-Detection\h5_files\ft_model_plants-10epochs.h5")
#pb_fname = "./my_model/saved_model.pb"

#with open("./vgg16_cat_list.pk", 'rb') as f:
#    cat_list = pk.load(f)

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def prepare_img_256(img_path):
    # urllib.request.urlretrieve(img_path, 'save.jpg')
    img = load_img(img_path, target_size=(200, 200))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) / 255
    return x



def pipe_loc(img_256, model):
    print("Determining damage plant")
    out = model.predict(img_256)
    pred_labels = np.argmax(out, axis=1)
    idx_to_classes = {0: 'Apple___Apple_scab',
                  1: 'Apple___Black_rot',
                  2: 'Apple___Cedar_apple_rust',
                  3: 'Apple___healthy',
                  4: 'Blueberry___healthy',
                  5: 'Cherry___Powdery_mildew',
                  6: 'Cherry___healthy',
                  7: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                  8: 'Corn___Common_rust',
                  9: 'Corn___Northern_Leaf_Blight',
                  10: 'Corn___healthy',
                  11: 'Grape___Black_rot',
                  12: 'Grape___Esca_(Black_Measles)',
                  13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  14: 'Grape___healthy',
                  15: 'Orange___Haunglongbing_(Citrus_greening)',
                  16: 'Peach___Bacterial_spot',
                  17: 'Peach___healthy',
                  18: 'Pepper,_bell___Bacterial_spot',
                  19: 'Pepper,_bell___healthy',
                  20: 'Potato___Early_blight',
                  21: 'Potato___Late_blight',
                  22: 'Potato___healthy',
                  23: 'Raspberry___healthy',
                  24: 'Soybean___healthy',
                  25: 'Squash___Powdery_mildew',
                  26: 'Strawberry___Leaf_scorch',
                  27: 'Strawberry___healthy',
                  28: 'Tomato___Bacterial_spot',
                  29: 'Tomato___Early_blight',
                  30: 'Tomato___Late_blight',
                  31: 'Tomato___Leaf_Mold',
                  32: 'Tomato___Septoria_leaf_spot',
                  33: 'Tomato___Spider_mites Two-spotted_spider_mite',
                  34: 'Tomato___Target_Spot',
                  35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                  36: 'Tomato___Tomato_mosaic_virus',
                  37: 'Tomato___healthy',
                  38: 'Background_without_leaves'}
    
    #print(out)
    print(pred_labels)
    for key in idx_to_classes.keys():
        if pred_labels[0] == key:
            print("The Model predicts the leaf entered Belongs to\n  {} ".format(idx_to_classes[key]))
            print("Plant and Disease detection complete.")
            #return idx_to_classes[key]
    #print("Plant and Disease detection complete.")
    return pred_labels




def pipe(img_path):
    img_256 = prepare_img_256(img_path)
    x = pipe_loc(img_256, model2)
       
    result = x

    return result




#pipe("D:\\Machine Learning Datsets\\PlantVillage\\PlantVillage\\val\\Potato___Early_blight\\2d149f7a-4b0a-40a6-8d0b-1d1f14e5e696___RS_Early.B 9143.JPG")
#pipe(r"C:\Users\Shreyas Desai\Documents\Plant-Disease-Detection\Plant_leave_diseases_dataset_with_augmentation\Cherry___Powdery_mildew\image (22).JPG")
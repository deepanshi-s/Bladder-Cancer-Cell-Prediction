import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,Callback
import PIL
import warnings
import os
import cv2
from io import StringIO
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from helper1 import trainGenerator
from unet_model import unet 



h1=unet((512,512,3))

path_of_training = '//content//drive//My Drive//abc_1//pqr_2'

data_gen_args = dict()

Generator = trainGenerator(1,path_of_training,'train_og','train_mask',data_gen_args,save_to_dir =  None)

h1.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('unet_final.hdf5', monitor='loss',verbose=1, save_best_only=True)

h1.fit_generator(Generator,steps_per_epoch=20,epochs=10,callbacks=[model_checkpoint])

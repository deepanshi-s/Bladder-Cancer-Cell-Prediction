from keras.utils import to_categorical
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
from keras.models import load_model

from helper1 import testGenerator
from unet_model import unet


h1=unet((512,512,3))
h1.load_weights('unet_membrane.hdf5')   #path to weights of unet model 

testGene = testGenerator("test")         #path to test folder 
results = h1.predict_generator(testGene,5,verbose=1)
print(results.shape)
#print(results.shape)

saveResult("test_results",results)       #path to save results

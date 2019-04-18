from numpy import *

import numpy as np
import csv
import imageio
import time
import sys
import os
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import pandas as pd

#####################################
patchDir = r'C:\Users\10250153\bacteria3\data\patch'  # path to the dir with annotated patches
patchDir = os.path.join(patchDir, '')
fname = os.path.join(patchDir, 'train', 'patch.log')
print(fname, end='\n\n')

# ys = np.array([], dtype=np.float32).reshape(0,5)# int64
# xs = np.array([[1,2,3,4,5],[10,20,30,40,50]])
# ys = np.vstack([ys, xs])
# print(ys)

# if True:
# #     print("exiting...")
# #     sys.exit()

#####################################
isFirst = True
count = 0

f = open(fname, 'r')
reader = csv.reader(f)
for row in reader:
    if not ''.join(row).startswith("#"):
        imgPath = os.path.join(patchDir, row[0])
        im = (imread(imgPath)).astype(float32)
        imHeight,imWidth = im.shape
        imSize = im.size
        if isFirst:
            dataX = np.array(im, dtype=np.float32).reshape(1, imSize)
            isFirst = False
        else:
            dataX1 = np.array(im, dtype=np.float32).reshape(1, imSize)
            dataX = np.concatenate([dataX,   dataX1])

        count = count + 1
        print('\n', count , ': ', row, '\n', imgPath, ' -> ', dataX.shape, end='\n\n')# '\ndata.type=', str(type(dataX)),

f.close()

#####################################
#

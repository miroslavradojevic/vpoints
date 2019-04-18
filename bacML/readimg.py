file_path = "C:\\Users\\10250153\\bacteria3\\generated.tif"

print("\nopencv-python")
import cv2
img1 = cv2.imread(file_path)
print("type = ", type(img1))
print("shape = ", img1.shape)
print("dtype = ", img1.dtype)

print("\nscikit-image")
from skimage import io
img2 = io.imread(file_path)
print("type = ", type(img2))
print("shape = ", img2.shape)
print("dtype = ", img2.dtype)

print("\nPIL")
from PIL import Image
import numpy
img3 = numpy.array(Image.open(file_path))
print("type = ", type(img3))
print("shape = ", img3.shape)
print("dtype = ", img3.dtype)

#from scipy.misc import imread

# import matplotlib.image as mpimg
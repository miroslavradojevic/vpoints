import cv2
import sys
import numpy as np
from scipy import ndimage
np.set_printoptions(threshold=sys.maxsize)

img = cv2.imread("C:\\Users\\miros\\stack\\vpoints\\images\\blobs.tif", cv2.IMREAD_GRAYSCALE)

img = ((img > 124).astype(int) * 255).astype(np.uint8)

labeled, nr_objects = ndimage.label(img)

refined_mask = img.copy()
for label in range(nr_objects):
    if np.sum(labeled == label + 1) < 200:
        refined_mask[labeled == label + 1] = 0

cv2.imwrite("C:\\Users\\miros\\stack\\vpoints\\images\\blobs_removed.tif", refined_mask)

if True:
    quit("done")

##############
img = cv2.imread("C:\\Users\\miros\\stack\\vpoints\\images\\blobs.jpg", cv2.IMREAD_GRAYSCALE)

#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 500

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

cv2.imwrite("C:\\Users\\miros\\stack\\vpoints\\images\\blobs_removed.jpg", img2)
import os
import sys
import cv2
import math
import numpy as np
from pathlib import Path
from geometry import cross

print(math.fabs(cross(1, 3, 4, 5)))
print(cross(4, 5, 1, 3))

if True:
    quit("quitting debug...")

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib
# import matplotlib.pyplot
# matplotlib.rcParams['interactive'] == True

MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 5
GAUSS_BLUR = 9

try:
    img_path = sys.argv[1]
    min_val_canny = int(sys.argv[2])
    max_val_canny = int(sys.argv[3])
    threshold_hough = int(sys.argv[4])
except:
    quit("Wrong command.\n"
         "Usage:\n"
         "python vpdetector.py image_path min_val_canny max_val_canny threshold_hough\n"
         "Example:\n"
         "python vpdetector.py 5D4L1L1D_L.jpg 100 200 100\n")

if not os.path.isfile(img_path) or not Path(img_path).suffix == '.jpg':
    quit("Error:", img_path, "must be a .jpg file")

# Read color image
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)


img_name = os.path.splitext(os.path.basename(img_path))[0]
print("Read image", img_name, "\t", type(img_color), "\tdimensions:", img_color.shape)


# Create directory with exported results
img_dir = os.path.dirname(img_path)
out_dir_name = "VPDET," + img_name

for arg_idx in range(2, len(sys.argv)):
    out_dir_name += "," + str(sys.argv[arg_idx])

out_dir_path = os.path.join(img_dir, out_dir_name)

if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)

h, w, l = img_color.shape

# Extend image
img_color = cv2.copyMakeBorder(img_color, int(h / 2), int(h / 2), int(w / 2), int(w / 2), cv2.BORDER_CONSTANT, value=[255, 255, 255])
cv2.imwrite(os.path.join(out_dir_path, img_name + "_input.jpg"), img_color)


# Convert to gray-scale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_uint8.jpg"), img_gray)

# Extend W x H image to double size 2W x 2H
# img_gray = cv2.copyMakeBorder(img_gray, int(h / 2), int(h / 2), int(w / 2), int(w / 2), cv2.BORDER_CONSTANT, value=[255, 255, 255])
# cv2.imwrite(os.path.join(out_dir_path, img_name + "_uint8_x2.jpg"), img_gray)

# Find the edges in the image using canny detector
img_edges = cv2.Canny(img_gray, min_val_canny, max_val_canny)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_before.jpg"), img_edges)

# Crop the border edges caused by the image extension

# Create a mask
mask = np.zeros(img_edges.shape, np.uint8)

p_tl = [math.ceil(h / 2) + 5, math.ceil(w / 2) + 5]
p_bl = [math.floor(1.5 * h) - 5, math.ceil(w / 2) + 5]
p_br = [math.floor(1.5 * h) - 5, math.floor(1.5 * w) - 5]
p_tr = [math.ceil(h / 2) + 5, math.floor(1.5 * w) - 5]

points = np.array([[p_tl, p_bl, p_br, p_tr]])
cv2.fillPoly(mask, points, 255)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_mask.jpg"), mask)

# Apply mask to original image
img_edges = cv2.bitwise_and(img_edges, img_edges, mask = mask)

cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_after.jpg"), img_edges)

# Remove small components
# #find all your connected components (white blobs in your image)
# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
# #connectedComponentswithStats yields every seperated component with information on each of them, such as size
# #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
# sizes = stats[1:, -1]; nb_components = nb_components - 1
#
# # minimum size of particles we want to keep (number of pixels)
# #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
# min_size = 150
#
# #your answer image
# img2 = np.zeros((output.shape))
# #for every component in the image, you keep it only if it's above min_size
# for i in range(0, nb_components):
#     if sizes[i] >= min_size:
#         img2[output == i + 1] = 255

# Blur the edge image
img_edges = cv2.GaussianBlur(img_edges, (GAUSS_BLUR, GAUSS_BLUR),0)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_blurred.jpg"), img_edges)

# Detect points that form a line
lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, threshold_hough, MAX_LINE_GAP, MIN_LINE_LENGTH)

if lines is None or len(lines) == 0:
    quit("error: no lines extracted")

# Draw lines on the image
# img_lines = img_color

l_min = math.inf
l_max = -math.inf
count = 0
for line in lines:
    x1, y1, x2, y2 = line[0]
    l = math.sqrt(pow(x1-x2, 2) + pow(y1-y2,2))
    l_min = l if l < l_min else l_min
    l_max = l if l > l_max else l_max
    count += 1
    cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 255), 2)

print("found", len(lines), "lines, with length range", "{0:.2f}".format(l_min), " - ", l_max)
print("count=", count)
# Export result
cv2.imwrite(os.path.join(out_dir_path, img_name + "_lines.jpg"), img_color)

def intersections(lines):
  print("Hello from a function")

#
for i in range(len(lines)):
  # print(x)
  my_function()
else:
  print("Finally finished!")







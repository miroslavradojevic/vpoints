import os
import sys
import cv2
import math
import numpy as np
from pathlib import Path
from geometry import cross
from geometry import modulus
from scipy import ndimage
import matplotlib.pyplot as plt

# print(math.fabs(cross(1, 3, 4, 5)))
# print(cross(4, 5, 1, 3))


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib
# import matplotlib.pyplot
# matplotlib.rcParams['interactive'] == True

MIN_LINE_LENGTH = 100
MIN_SIZE_CONN_COMPONENT = 100
MAX_LINE_GAP = 5
GAUSS_BLUR = 9
SMALL_POSITIVE_INTENSITY = 0.0001
MS_RADIUS2 = 5 * 5
MS_MAXITER = 10
MS_EPSILON2 = 0.0001


############################################################

def get_angles(lines, nr_angles):
    for i in range(len(lines)):
        px, py, rx, ry = lines[i][0]
        rx = rx - px
        ry = ry - py
        r_mod = modulus(rx, ry)
        alpha = math.floor(math.acos(rx / r_mod) / (math.pi / nr_angles))
        print(alpha)


# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
def get_intersections(lines):
    intersection_coord = []
    intersection_weight = []
    count = 0

    lines_count = len(lines)

    for i in range(lines_count):
        px, py, rx, ry = lines[i][0]  # read line endpoints
        rx = rx - px
        ry = ry - py
        r_mod = modulus(rx, ry)

        for j in range(i + 1, lines_count):
            qx, qy, sx, sy = lines[j][0]
            sx = sx - qx
            sy = sy - qy
            s_mod = modulus(sx, sy)

            r_x_s = cross(rx, ry, sx, sy)

            if math.fabs(r_x_s) > SMALL_POSITIVE_INTENSITY:

                # compute location
                qpx = qx - px
                qpy = qy - py

                qp_x_s = cross(qpx, qpy, sx, sy)
                t = qp_x_s / r_x_s

                qp_x_r = cross(qpx, qpy, rx, ry)
                u = qp_x_r / r_x_s

                if 0 <= t <= 1 or 0 <= u <= 1:
                    continue

                isec_x = px + t * rx  # intersection
                isec_y = py + t * ry

                # isec2_x = qx + u * sx # intersection ver. 2
                # isec2_y = qy + u * sy

                # add if it is within the boundaries of the extended image
                if 0 <= isec_x < w_ext and 0 <= isec_y < h_ext:
                    count += 1
                    intersection_coord.append([isec_x, isec_y])
                    intersection_weight.append(r_mod + s_mod)

    return intersection_coord, intersection_weight


# mean-shift (non-blurring) uses neighbourhood defined with radius
def mean_shift(in_xy, radius2, maxiter, epsilon2):
    out_xy = in_xy.copy()

    conv = np.array([0.0, 0.0], dtype=float)
    next = np.array([0.0, 0.0], dtype=float)

    for i in range(len(out_xy)):
        sys.stdout.write("%f%%   \r" % (100 * (i + 1) / len(out_xy)))
        sys.stdout.flush()
        # refine in_xy[i] location and store the result in out_xy[i]
        conv[0] = in_xy[i][0]
        conv[1] = in_xy[i][1]
        iter = 0
        while True:
            cnt = 0
            next[0] = 0  # // local mean is the follow-up location
            next[1] = 0

            for j in range(len(in_xy)):
                x2 = math.pow(in_xy[j][0] - conv[0], 2)
                if x2 <= radius2:
                    y2 = math.pow(in_xy[j][1]-conv[1], 2)
                    if x2 + y2 <= radius2:
                        next[0] += in_xy[j][0]
                        next[1] += in_xy[j][1]
                        cnt += 1

            next[0] /= cnt
            next[1] /= cnt

            d2 = math.pow(next[0] - conv[0], 2) + math.pow(next[1] - conv[1], 2)

            conv[0] = next[0]
            conv[1] = next[1]

            iter += 1

            if iter >= maxiter or d2 <= epsilon2:
                break
        print("iter=", iter, "d2=", d2)

        out_xy[i][0] = conv[0]
        out_xy[i][1] = conv[1]

    return out_xy


############################################################
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
         "python vpdetector.py 5D4L1L1D_L.jpg 100 200 100\n"
         "python vpdetector.py C:\\Users\\miros\\stack\\vpoints\\images\\5D4L1L1D_L.jpg 100 300 100\n")

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

# Extend color image (2x original size)
img_color = cv2.copyMakeBorder(img_color, int(h / 2), int(h / 2), int(w / 2), int(w / 2), cv2.BORDER_CONSTANT,
                               value=[255, 255, 255])
cv2.imwrite(os.path.join(out_dir_path, img_name + "_input.jpg"), img_color)

h_ext, w_ext, l_ext = img_color.shape

# Convert to gray-scale
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_uint8.jpg"), img_gray)

# Find the edges in the image using canny detector
img_edges = cv2.Canny(img_gray, min_val_canny, max_val_canny)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_before.jpg"), img_edges)
del img_gray  # release memory

# Crop the border edges caused by the image extension
mask = np.zeros(img_edges.shape, np.uint8)

p_tl = [math.ceil(h / 2) + 5, math.ceil(w / 2) + 5]
p_bl = [math.floor(1.5 * h) - 5, math.ceil(w / 2) + 5]
p_br = [math.floor(1.5 * h) - 5, math.floor(1.5 * w) - 5]
p_tr = [math.ceil(h / 2) + 5, math.floor(1.5 * w) - 5]

points = np.array([[p_tl, p_bl, p_br, p_tr]])
cv2.fillPoly(mask, points, 255)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_mask.jpg"), mask)

# Apply mask to original image
img_edges = cv2.bitwise_and(img_edges, img_edges, mask=mask)
del mask  # release memory

cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_after.jpg"), img_edges)

img_edges = ((img_edges > 124).astype(int) * 255).astype(np.uint8)  # binary image: 0 and 255 values only

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_edges, connectivity=8)
sizes = stats[1:, -1]
nb_components = nb_components - 1

img2 = np.zeros(img_edges.shape, dtype=np.uint8)

# Alternative (slower) connected components implementation
# labeled, nr_objects = ndimage.label(img_edges)
# for label in range(nr_objects):
#     sys.stdout.write("pruning components: %d%%   \r" % (100 * (label + 1) / nr_objects))
#     sys.stdout.flush()
#     if np.sum(labeled == label + 1) >= MIN_SIZE_CONN_COMPONENT:
#         img2[labeled == label + 1] = 255

for i in range(nb_components):
    sys.stdout.write("pruning c-components: %d%%   \r" % (100 * (i + 1) / nb_components))
    sys.stdout.flush()
    if sizes[i] >= MIN_SIZE_CONN_COMPONENT:
        img2[output == i + 1] = 255

img_edges = img2
del img2

cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_after_refined.jpg"), img_edges)

# Blur the edge image
img_edges = cv2.GaussianBlur(img_edges, (GAUSS_BLUR, GAUSS_BLUR), 0)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_blurred.jpg"), img_edges)

# Detect points that form a line
lines = cv2.HoughLinesP(img_edges, 1, np.pi / 90, threshold_hough, MAX_LINE_GAP, MIN_LINE_LENGTH)

if lines is None or len(lines) == 0:
    quit("error: no lines extracted")

# Draw lines on the image
viz_lines = img_color.copy()

l_min, l_max = [math.inf, -math.inf]
x_min, x_max = [math.inf, -math.inf]
y_min, y_max = [math.inf, -math.inf]

for line in lines:
    x1, y1, x2, y2 = line[0]

    l = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    l_min = min(l_min, l)
    l_max = max(l_max, l)

    x_min = min(x_min, min(x1, x2))
    x_max = max(x_max, max(x1, x2))

    y_min = min(y_min, min(y1, y2))
    y_max = max(y_max, max(y1, y2))

    cv2.line(viz_lines, (x1, y1), (x2, y2), (0, 255, 255), 2)

print(len(lines), "lines, with length range", "{0:.2f}".format(l_min), " - ", "{0:.2f}".format(l_max))
# print("x range", "{0:.2f}".format(x_min), " - ", "{0:.2f}".format(x_max))
# print("y range", "{0:.2f}".format(y_min), " - ", "{0:.2f}".format(y_max))

# Export result
cv2.imwrite(os.path.join(out_dir_path, img_name + "_lines.jpg"), viz_lines)
del viz_lines  # release memory

#######################################
Ixy, Iw = get_intersections(lines)
print(len(Ixy), "intersections found")

#######################################
# visualize intersection points
viz_isec = img_color

for i in range(len(Ixy)):
    x = int(Ixy[i][0])
    y = int(Ixy[i][1])
    cv2.circle(viz_isec, (x, y), int(10), (0, 0, 255), 5)

cv2.imwrite(os.path.join(out_dir_path, img_name + "_intersections.jpg"), viz_isec)
del viz_isec  # release memory


#######################################
# mean-shift intersection points to converge towards vanishing point candidates
# print("mean-shift...")
# Jxy = mean_shift(Ixy, MS_RADIUS2, MS_MAXITER, MS_EPSILON2)
# print("\ndone")

#######################################
# create cumulative image of intersection locations




#######################################
# cluster converged points




#######################################
# pick vanishing points



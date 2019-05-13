import os
import sys
import cv2
import math
import time
import numpy as np
from pathlib import Path

RESIZE_WIDTH = 512
RESIZE_HEIGHT = 512
MIN_LINE_LENGTH = 50
MIN_SIZE_CONN_COMPONENT = 40
MAX_LINE_GAP = 5
GAUSS_BLUR = 9
SMALL_POSITIVE_INTENSITY = 1e-6
MS_D_XY = 20
MS_MAXITER = 1000
MS_EPSILON2 = 1e-6
CLUSTER_DIST = 1
CLUSTER_MIN_COUNT = 10
GAUSSIAN_ANGLE_MEAN_DEG = 45
GAUSSIAN_ANGLE_STD_DEG = 20


############################################################
# function definitions
############################################################
def cross(v1x, v1y, v2x, v2y):
    return v1x * v2y - v1y * v2x


def modulus(vx, vy):
    return math.sqrt(vx * vx + vy * vy)


def dot(v1x, v1y, v2x, v2y):
    return v1x * v2x + v1y * v2y


def get_angles(lines, nr_angles):
    for i in range(len(lines)):
        px, py, rx, ry = lines[i][0]
        rx = rx - px
        ry = ry - py
        r_mod = modulus(rx, ry)
        alpha = math.floor(math.acos(rx / r_mod) / (math.pi / nr_angles))
        print(alpha)


def angle_gaussian_sensitivity(input_angle_deg, mean_deg, sigma_deg):
    return np.exp(-(input_angle_deg - mean_deg) ** 2 / (2 * sigma_deg ** 2))


def get_intersections(lines, angle_mean, angle_std):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    intersection_coord = []
    intersection_weight = []
    intersection_angle = []
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

                p_isec_x = px - isec_x
                p_isec_y = py - isec_y
                p_isec_mod = modulus(p_isec_x, p_isec_y)

                r_isec_x = (px + rx) - isec_x
                r_isec_y = (py + ry) - isec_y
                r_isec_mod = modulus(r_isec_x, r_isec_y)

                q_isec_x = qx - isec_x
                q_isec_y = qy - isec_y
                q_isec_mod = modulus(q_isec_x, q_isec_y)

                s_isec_x = (qx + sx) - isec_x
                s_isec_y = (qy + sy) - isec_y
                s_isec_mod = modulus(s_isec_x, s_isec_y)

                isec_angle_deg = math.acos(
                    dot(p_isec_x / p_isec_mod, p_isec_y / p_isec_mod, q_isec_x / q_isec_mod, q_isec_y / q_isec_mod)) * (
                                             180.0 / math.pi)

                # add if it is within the boundaries of the extended image
                if 0 <= isec_x < w_ext and 0 <= isec_y < h_ext:
                    count += 1
                    intersection_coord.append([isec_x, isec_y])
                    intersection_angle.append(isec_angle_deg)
                    intersection_weight_value = angle_gaussian_sensitivity(isec_angle_deg, angle_mean, angle_std)
                    intersection_weight_value *= (r_mod / max(p_isec_mod, r_isec_mod)) * (
                                s_mod / max(q_isec_mod, s_isec_mod))
                    intersection_weight.append(intersection_weight_value)

    return intersection_coord, intersection_weight, intersection_angle


def mean_shift(input_image, d_xy, maxiter, epsilon2):
    input_image_height = input_image.shape[0]
    input_image_width = input_image.shape[1]

    input_image_list = input_image.tolist()  # retrieving values works way faster if used as list

    ms_xy = []
    for x in range(input_image_height):
        for y in range(input_image_width):

            if input_image_list[x][y] > 0:

                conv_x = x
                conv_y = y
                iteration = 0

                while True:
                    cnt = 0
                    next_x = 0  # local mean represents the follow-up location
                    next_y = 0

                    for x1 in range(x - d_xy, x + d_xy + 1):
                        for y1 in range(y - d_xy, y + d_xy + 1):
                            if 0 <= x1 < input_image_height and 0 <= y1 < input_image_width:
                                val = input_image_list[x1][y1]
                                if val > 0:
                                    next_x += val * x1
                                    next_y += val * y1
                                    cnt += val

                    next_x /= cnt
                    next_y /= cnt

                    d2 = math.pow(next_x - conv_x, 2) + math.pow(next_y - conv_y, 2)

                    conv_x = next_x
                    conv_y = next_y

                    iteration += 1

                    if iteration >= maxiter or d2 <= epsilon2:
                        break

                ms_xy.append([conv_x, conv_y])

    del input_image_list
    return ms_xy


def clustering(p_xy, cluster_dist):
    cluster_dist2 = math.pow(cluster_dist, 2)

    ####
    nbridx = []
    for i in range(len(p_xy)):
        nbridx.append([])

    nr_points = len(p_xy)
    ####

    for i in range(nr_points):
        for j in range(i + 1, nr_points):
            cd = math.pow(p_xy[i][0] - p_xy[j][0], 2) + math.pow(p_xy[i][1] - p_xy[j][1], 2)
            if cd <= cluster_dist2:
                nbridx[i].append(j)
                nbridx[j].append(i)

    ####
    labels = np.zeros(nr_points, dtype=np.int32)
    for i in range(nr_points):
        labels[i] = i

    ####
    for i in range(nr_points):
        for nbri in range(len(nbridx[i])):
            j = nbridx[i][nbri]
            if labels[j] != labels[i]:
                curr_label = labels[j]
                new_label = labels[i]

                labels[j] = new_label

                # set all that were curr_label to new_label
                for k in range(nr_points):
                    if labels[k] == curr_label:
                        labels[k] = new_label

    return labels


def extract(labels, locs_xy, min_count):
    nr_locs = len(labels)

    checked = [False] * nr_locs

    out = []

    for i in range(nr_locs):
        if not checked[i]:
            centroid_x = locs_xy[i][0]
            centroid_y = locs_xy[i][1]
            count = 1
            checked[i] = True
            for j in range(i + 1, nr_locs):
                if not checked[j]:
                    if labels[j] == labels[i]:
                        centroid_x += locs_xy[j][0]
                        centroid_y += locs_xy[j][1]
                        count += 1
                        checked[j] = True

            if count >= min_count:
                out.append([centroid_x / count, centroid_y / count, count])

    return out


############################################################
# programme
############################################################
try:
    img_path = sys.argv[1]
    min_val_canny = int(sys.argv[2])
    max_val_canny = int(sys.argv[3])
    threshold_hough = int(sys.argv[4])
except:
    quit("Wrong command.\n"
         "Usage:\n"
         "python vpdetector.py P1  P2  P3  P4  P5\n"
         "----------------------------------------------------\n"
         "Parameter legend:\n"
         "P1 = image file name or full path (accepts only jpg extension)\n"
         "P2 = canny edge detector: lower threshold, in [0, inf]\n"
         "P3 = canny edge detector: upper threshold, in [0, inf]\n"
         "P4 = hough_threshold, in [0, inf]\n"
         "P5 = score_threshold, in [0, 1]\n"
         "----------------------------------------------------\n"
         "Example calls:\n"
         "cd directory_containing_image; python vpdetector.py 5D4L1L1D_L.jpg 100 300 100 0.8\n"
         "python vpdetector.py C:\\Users\\miros\\stack\\vpoints\\images\\5D4L1L1D_L.jpg 100 300 100 0.8\n")

if not os.path.isfile(img_path) or not Path(img_path).suffix == '.jpg':
    quit("Error:", img_path, "must be a .jpg file")

############################################################
# Read color image
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)

img_name = os.path.splitext(os.path.basename(img_path))[0]
print("Read image", img_name, type(img_color), "dimensions:", img_color.shape)

# Resize input image
img_color = cv2.resize(img_color, (RESIZE_WIDTH, RESIZE_HEIGHT))
print("Resized", img_name, type(img_color), "dimensions:", img_color.shape)

# Create directory with exported results
img_dir = os.path.dirname(img_path)
out_dir_name = "VPDET_" + time.strftime("%Y%m%d-%H%M%S") + "_" + img_name

for arg_idx in range(2, len(sys.argv)):
    out_dir_name += "_" + str(sys.argv[arg_idx])

out_dir_path = os.path.join(img_dir, out_dir_name)

if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)

h, w, l = img_color.shape

# Extend input image field (2x original size)
img_color = cv2.copyMakeBorder(img_color, int(h / 2), int(h / 2), int(w / 2), int(w / 2), cv2.BORDER_CONSTANT,
                               value=[255, 255, 255])
cv2.imwrite(os.path.join(out_dir_path, img_name + "_input.jpg"), img_color)

h_ext, w_ext, l_ext = img_color.shape

# Convert to gray-scale for further processing
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_uint8.jpg"), img_gray)

############################################################
# Find the edges in the image using canny detector
img_edges = cv2.Canny(img_gray, min_val_canny, max_val_canny)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_edges_canny_before.jpg"), img_edges)

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

############################################################
# https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_edges, connectivity=8)
sizes = stats[1:, -1]
nb_components = nb_components - 1

img2 = np.zeros(img_edges.shape, dtype=np.uint8)

# Alternative (slower) connected components implementation
# labeled, nr_objects = ndimage.label(img_edges)
# for label in range(nr_objects):
#     if np.sum(labeled == label + 1) >= MIN_SIZE_CONN_COMPONENT:
#         img2[labeled == label + 1] = 255

for i in range(nb_components):
    sys.stdout.write("pruning c-components: %d%%   \r" % (100 * (i + 1) / nb_components))
    sys.stdout.flush()
    if sizes[i] >= MIN_SIZE_CONN_COMPONENT:
        img2[output == i + 1] = 255

img_edges = img2
del img2 # release memory

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
Ixy, Iw, Iang = get_intersections(lines, GAUSSIAN_ANGLE_MEAN_DEG, GAUSSIAN_ANGLE_STD_DEG)
print(len(Ixy), "intersections found", flush=True)
print("weight:", np.amin(Iw), " -- ", np.amax(Iw), flush=True)
print("angles:", np.amin(Iang), " -- ", np.amax(Iang), flush=True)

#######################################
# visualize intersection points
viz_isec = img_color.copy()

for i in range(len(Ixy)):
    x = int(Ixy[i][0])
    y = int(Ixy[i][1])
    cv2.circle(viz_isec, (x, y), int(5), (0, 0, 255), 2)

cv2.imwrite(os.path.join(out_dir_path, img_name + "_intersections.jpg"), viz_isec)
del viz_isec  # release memory

#######################################
# create cumulative map of intersection locations with their weights
img_intersec = np.zeros(img_edges.shape, dtype=np.float32)

for lineIdx in range(len(Ixy)):
    xIdx = int(math.floor(Ixy[lineIdx][0]))
    yIdx = int(math.floor(Ixy[lineIdx][1]))
    img_intersec[yIdx, xIdx] += Iw[lineIdx]
    sys.stdout.write("intersection %d / %d:\tx=%f[%d], y=%f[%d], w=%f\r" % (
        (lineIdx + 1), len(Ixy), Ixy[lineIdx][0], xIdx, Ixy[lineIdx][1], yIdx, Iw[lineIdx]))
    sys.stdout.flush()

print("\naccumulated ", np.sum(img_intersec > 0), " | ", len(Ixy), " ", (100.0 * np.sum(img_intersec > 0) / len(Ixy)),
      "%")

img_intersec_min = np.amin(img_intersec)
img_intersec_max = np.amax(img_intersec)

# min-max normalize between 0 and 255 before exporting
img_intersec = 255 * ((img_intersec - img_intersec_min) / (img_intersec_max - img_intersec_min))
cv2.imwrite(os.path.join(out_dir_path, img_name + "_img_intersec.jpg"), img_intersec)

print("\nmin = %f max = %f" % (np.amin(img_intersec), np.amax(img_intersec)), flush=True)

########################################################################################################
print("plot intersections...", end=" ", flush=True)
img_intersec_dimensions = list(img_intersec.shape)
img_intersec_dimensions.append(3)  # color image needed for color plots

img_intersec_viz = np.zeros(tuple(img_intersec_dimensions), dtype=np.uint8)
img_intersec_viz[:, :, 0] = img_intersec.astype(np.uint8)
img_intersec_viz[:, :, 1] = img_intersec.astype(np.uint8)
img_intersec_viz[:, :, 2] = img_intersec.astype(np.uint8)
for lineIdx in range(len(Ixy)):
    xIdx = int(math.floor(Ixy[lineIdx][0]))
    yIdx = int(math.floor(Ixy[lineIdx][1]))
    cv2.circle(img_intersec_viz, (xIdx, yIdx), int(8), (0, 255, 0), 1)
cv2.imwrite(os.path.join(out_dir_path, img_name + "_Ixy.jpg"), img_intersec_viz)
del img_intersec_viz
print("done.", flush=True)

########################################################################################################
print("mean shift...", end=" ", flush=True)

start_time = time.time()

p_xy = mean_shift(img_intersec, MS_D_XY, MS_MAXITER,
                  MS_EPSILON2)  # TODO output weights too w_xy that correspond to each location

print("done. %d locations, %ssec" % (len(p_xy), time.time() - start_time), flush=True)

########################################################################################################
print("plot converged intersections...", end=" ", flush=True)
img_intersec_viz = np.zeros(tuple(img_intersec_dimensions), dtype=np.uint8)
img_intersec_viz[:, :, 0] = img_intersec.astype(np.uint8)
img_intersec_viz[:, :, 1] = img_intersec.astype(np.uint8)
img_intersec_viz[:, :, 2] = img_intersec.astype(np.uint8)

for i in range(len(p_xy)):
    cv2.circle(img_intersec_viz, (int(p_xy[i][1]), int(p_xy[i][0])), int(1), (0, 255, 255), -1)

cv2.imwrite(os.path.join(out_dir_path, img_name + "_Pxy.jpg"), img_intersec_viz)
del img_intersec_viz
print("done.", flush=True)

#######################################
print("clustering...", end=" ", flush=True)
start_time = time.time()
lab = clustering(p_xy, CLUSTER_DIST)
print("done. %ssec" % (time.time() - start_time), flush=True)

#######################################
print("extract...", end=" ", flush=True)
clusters = extract(lab, p_xy, CLUSTER_MIN_COUNT)
print("done.", flush=True)
print(len(clusters), "clusters found", flush=True)
clusters.sort(key=lambda x: x[2], reverse=True)

#######################################

vpoints_viz = np.zeros(tuple(img_intersec_dimensions), dtype=np.uint8)
vpoints_viz[:, :, 0] = img_gray.astype(np.uint8)
vpoints_viz[:, :, 1] = img_gray.astype(np.uint8)
vpoints_viz[:, :, 2] = img_gray.astype(np.uint8)
for i in range(len(clusters)):
    # print(clusters[i], flush=True)
    plot_col = int(round(clusters[i][1]))
    plot_row = int(round(clusters[i][0]))
    cv2.circle(vpoints_viz, (plot_col, plot_row), int(round(math.sqrt(clusters[i][2]) / math.pi)), (0, 0, 255), -1)

cv2.imwrite(os.path.join(out_dir_path, img_name + "_vpoints.jpg"), vpoints_viz)
del vpoints_viz
del img_gray

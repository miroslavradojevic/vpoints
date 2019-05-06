from PIL import Image
import numpy as np
import random
from geometry import *
import cv2
import os

K = 8

W = 1024
H = 512
imgarray = np.zeros([H, W], dtype=np.single)

print(imgarray.shape)
# x - row, y - column
px = 50  # random.randint(0, H-1)
py = 50  # random.randint(0, W-1)
rx = 300 # random.randint(0, H-1)
ry = 300 # random.randint(0, W-1)
vx, vy = direction(px, py, rx, ry)

# print("p=(", px, ",", py, ")")
# print("r=(", rx, ",", ry, ")")
# print("v=[", vx, ",", vy, "]")

# viz_img = imgarray # .astype('uint8')
# cv2.circle(viz_img, (px, py), int(10), (0, 0, 255), 4)
# cv2.imwrite("gen_img_line.jpg", viz_img)

for x in range(0, H):
    for y in range(0, W):
        s1 = math.pow(weight(x, y, px, py, -vx, -vy), K)
        s2 = math.pow(weight(x, y, rx, ry, +vx, +vy), K)
        imgarray[x, y] += max(s1, s2)  # * 255  # ((1.0 * x) / (H - 1)) * 255


px = 50
py = 300
rx = 250
ry = 300
vx, vy = direction(px, py, rx, ry)

for x in range(0, H):
    for y in range(0, W):
        s1 = math.pow(weight(x, y, px, py, -vx, -vy), K)
        s2 = math.pow(weight(x, y, rx, ry, +vx, +vy), K)
        imgarray[x, y] += max(s1, s2)


px = 55
py = 55
rx = 250
ry = 250
vx, vy = direction(px, py, rx, ry)

for x in range(0, H):
    for y in range(0, W):
        s1 = math.pow(weight(x, y, px, py, -vx, -vy), K)
        s2 = math.pow(weight(x, y, rx, ry, +vx, +vy), K)
        imgarray[x, y] += max(s1, s2)

# print(imgarray.max(axis=0))
img_max = np.amax(imgarray)
img_min = np.amin(imgarray)

imgarray = (imgarray - img_min)/(img_max - img_min)
imgarray *= 255
im = Image.fromarray(imgarray.astype('uint8')) # .convert('RGBA')
im.save('gen_img.tif')
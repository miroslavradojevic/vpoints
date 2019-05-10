import numpy as np
import cv2

# Create a black image
# img = np.zeros((512, 512, 3), dtype=np.uint8)
img = np.zeros((512, 512, 3), dtype=np.float)
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.circle(img, (256, 256), int(20), (0, 255, 0), 1)
cv2.imwrite("draw_elements.jpg", img)

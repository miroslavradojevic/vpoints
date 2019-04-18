import numpy as np
from skimage import io
from matplotlib import pyplot as plt

a = np.random.rand(512, 512) * 255  # uint8 range 0-255
b = np.around(a).astype('uint8')  # round float values and cast as uint8
io.imsave('b.tif', b)
io.imshow(b)
plt.show()

c = np.ndarray(shape=(64, 64), dtype='uint8') * 0

for row in range(0, 64):
    for col in range(0, 64):
        c[row, col] = 0 if row < 32 else 255

io.imsave('c.tif', c)
io.imshow(c)
plt.show()

import sys
import  os
import random
from skimage import io
# from skimage.viewer import ImageViewer


image_path = "";

try:
    image_path = sys.argv[1]
except:
    print('Please pass directory_name')

if not os.path.isfile(image_path):
    print(image_path, " is not a file.")
    quit()

if not image_path.endswith(".tif"):
    print(image_path, " needs to be tif (stack).")

img_uint8 = io.imread(image_path)

print("Image stack loaded: " + str(img_uint8.shape) + " " + str(type(img_uint8)))

z = 0 #random.randint(0, img_uint8.shape[0])
x = 0 #random.randint(0, img_uint8.shape[1])
y = 0 #random.randint(0, img_uint8.shape[2])
D = 5
print("Extract patch at: x = " + str(x) + ", y = " + str(y) + ", z = " + str(z))

ima = img_uint8[z:z+1, x:x+5, y:y+5]
print(ima.shape)
print(ima)

print("Extract layer z = 0")

ima = img_uint8[0, :, :]
print(ima.shape)
print(ima)

print("Extract layer z = ", (img_uint8.shape[0]-1))

ima = img_uint8[(img_uint8.shape[0]-1), :, :]
print(ima.shape)
print(ima)

# viewer = ImageViewer(img_uint8)
# viewer.show()


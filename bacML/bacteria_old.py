import os # os.path.expanduser('~')
import sys
import csv
import numpy as np

from skimage import io
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Conv2D, Convolution3D, MaxPooling2D, MaxPooling3D, ZeroPadding3D
from keras.constraints import maxnorm
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

K.set_image_data_format("channels_first")
print("image_dim_ordering = ", K.image_dim_ordering())
print("image_data_format = ", K.image_data_format())

#########################################################################
# Load the data
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    mapping = {}

    category = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".tif")]

        # adding an early stop for sake of speed
        stop = 1
        for f in file_names:

            img_uint8 = io.imread(f)
            # img_flo32 = img_uint8.astype('float32')

            images.append(img_uint8)

            if d not in mapping:
                mapping[d] = category

            labels.append(category)

            # remove this to use full data set
            # if stop >= 10:
            #     break
            # stop += 1
            # end early stop

        category += 1

    return images, labels, mapping

# create CNN model
def createCNNModel(num_classes):
    print(img_len)
    print(img_wdt)
    print(img_hgt)

    if K.image_data_format() == 'channels_first':
        print("it was channels_first")
        # input_shape = (L, W, H)
    else:
        print("it was NOT channels_first")
        # input_shape = (W, H, L)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_len, img_hgt, img_wdt), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 50  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print(model.summary())
    return model, epochs
#########################################################################

directory_name = ""
try:
    directory_name = sys.argv[1]
except:
    print('Please pass directory_name')

if not os.path.isdir(directory_name):
    print(directory_name, " is not a directory.");
    quit();

# read dimensions from the first file of the train log
train_logfile = os.path.join(directory_name, 'train', 'patch.log')
print(train_logfile, end='\n')

# dimensions (read from the first tif image of the train dir)
img_width, img_height, img_length = np.nan, np.nan, np.nan
f = open(train_logfile, 'r')
reader = csv.reader(f)
for row in reader:
    if not ''.join(row).startswith("#"):
        image_path = os.path.join(directory_name, 'train', row[0], row[1])
        print('systematic_resampling 3D patch path = {}'.format(image_path))
        image_uint8 = io.imread(image_path)
        # image_flo32 = image_uint8.astype('float32')
        img_len = image_uint8.shape[0] # must be channels_first
        img_hgt = image_uint8.shape[1]
        img_wdt = image_uint8.shape[2]
        if True:
            break;
f.close()

data_dir_train = os.path.join(directory_name, 'train')
data_dir_validation = os.path.join(directory_name, 'validate')
data_dir_test = os.path.join(directory_name, 'test')

print("l={}, w={}, h={}\ntrain={}\nvalidation={}\ntest={}".format(img_len, img_wdt, img_hgt, data_dir_train, data_dir_validation, data_dir_test))

x_train, y_train, map_train = load_data(data_dir_train)
x_vld, y_vld, map_vld = load_data(data_dir_validation)
x_tst, y_tst, map_tst = load_data(data_dir_test)

print("finished reading datasets:")
print("{} train images".format(len(x_train)))
print(map_train)

print("{} validation images".format(len(x_vld)))
print(map_vld)

print("{} test images".format(len(x_tst)))
print(map_tst)

# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(x_train).astype('float32')
X_train = X_train / 255.0
#
X_vld = np.array(x_vld).astype('float32')
X_vld = X_vld / 255.0
#
X_tst = np.array(x_tst).astype('float32')
X_tst = X_tst / 255.0

# one hot encode outputs
y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train)
#
y_vld = np.array(y_vld)
y_vld = np_utils.to_categorical(y_vld)
#
y_tst = np.array(y_tst)
y_tst = np_utils.to_categorical(y_tst)

num_classes = len(map_train)

print("Data normalized and hot encoded.", num_classes, " classes.")

# create CNN model
model, epochs = createCNNModel(num_classes)
print("2D CNN Model created.")

# create CNN 3D model


# fit and run our model
seed = 7
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_vld, y_vld), nb_epoch=10, batch_size=64)

print("model computed!")

# Final evaluation of the model
scores = model.evaluate(X_tst, y_tst, batch_size=64, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred = model.predict(X_tst, batch_size=64) # compare this one with the y_tst
# compute P, R, F
# comparing y_tst (gold standard) and y_pred

print("done")

if True:
    sys.exit(0)

# nb_train_samples = 5000
# nb_validation_samples = 1000
# batch_size = 16

#--------------------------------------------------------------------
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))# (41,41,23)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu')) # alternative
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#--------------------------------------------------------------------

np.random.seed(133)
pixel_depth = 255.0  # Number of levels per pixel.
directory_name = ""
num_classes = None

classes = []

for root, dirs, files in os.walk(directory_name):
    for name in dirs:
        classes.append(os.path.join(directory_name, name))

num_classes = len(classes)

image_list = []

for root, dirs, files in os.walk(directory_name):
    for name in files:
        if name.endswith(".png"):
            image_list.append(os.path.join(directory_name, name))

for c in classes:

    print('\n' + c, end='\n')

    cImg = []
    for root, dirs, files in os.walk(c):
        for name in files:
            if name.endswith(".png"):
                cImg.append(os.path.join(c, name))

    print(len(cImg), " png files.", end='\t')

    # cIdx = np.random.randint(1, len(cImg) + 1)
    #
    # img = mpimg.imread(str(cImg[cIdx]))
    #
    # print(img.shape)
    #
    #
    # imgplot = plt.imshow(img, cmap="gray")
    # plt.show()

# augmentation configuration used for training
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# def load_letter(folder, min_num_images):
#     """Load the data for a single letter label."""
#     image_files = os.listdir(folder)
#     dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
#                          dtype=np.float32)
#     print(folder)
#     num_images = 0
#     for image in image_files:
#         image_file = os.path.join(folder, image)
#         try:
#             image_data = (imageio.imread(image_file).astype(float) -
#                           pixel_depth / 2) / pixel_depth
#             if image_data.shape != (image_size, image_size):
#                 raise Exception('Unexpected image shape: %s' % str(image_data.shape))
#             dataset[num_images, :, :] = image_data
#             num_images = num_images + 1
#         except (IOError, ValueError) as e:
#             print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
#
#     dataset = dataset[0:num_images, :, :]
#     if num_images < min_num_images:
#         raise Exception('Many fewer images than expected: %d < %d' %
#                         (num_images, min_num_images))
#
#     print('Full dataset tensor:', dataset.shape)
#     print('Mean:', np.mean(dataset))
#     print('Standard deviation:', np.std(dataset))
#     return dataset
#
#
# def maybe_pickle(data_folders, min_num_images_per_class, force=False):
#     dataset_names = []
#     for folder in data_folders:
#         set_filename = folder + '.pickle'
#         dataset_names.append(set_filename)
#         if os.path.exists(set_filename) and not force:
#             # You may override by setting force=True.
#             print('%s already present - Skipping pickling.' % set_filename)
#         else:
#             print('Pickling %s.' % set_filename)
#             dataset = load_letter(folder, min_num_images_per_class)
#             try:
#                 with open(set_filename, 'wb') as f:
#                     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
#             except Exception as e:
#                 print('Unable to save data to', set_filename, ':', e)
#
#     return dataset_names
#
# train_datasets = maybe_pickle(train_folders, 45000)
# test_datasets = maybe_pickle(test_folders, 1800)
import os
import sys
from skimage import io
import numpy as np
import random
import json
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

# python bacteria_identification.py directory_path D Ntrain Nvalid Ntest recompute
# expected location of the annotations and regions
#

##########################################################
def load_data(data_dir):
    ftifs = [os.path.join(data_dir, f)
             for f in os.listdir(data_dir)
             if f.endswith(".tif")]

    flist = []
    for ftif in ftifs:
        ftif_dir = os.path.abspath(os.path.join(ftif, os.pardir))
        ftif_name = os.path.basename(ftif)

        flist.append((
            ftif,
            os.path.join(ftif_dir, "annot", ftif_name),
            os.path.join(ftif_dir, "ml_region", "train", ftif_name),
            os.path.join(ftif_dir, "ml_region", "validate", ftif_name),
            os.path.join(ftif_dir, "ml_region", "test", ftif_name)
        ))
    return flist


##########################################################
def sample(wgts, mask, nsamp):
    cws = np.zeros(len(wgts))
    for i in range(len(cws)):
        if mask is None:
            cws[i] = wgts[i] + (0.0 if (i == 0) else cws[i - 1])
        else:
            cws[i] = (wgts[i] if (mask[i] == 255) else 0.0) + (0.0 if (i == 0) else cws[i - 1])
    out = np.zeros(nsamp).astype(int)
    totalmass = cws[len(cws) - 1]
    # systematic re-sampling
    i = int(0)
    u1 = (totalmass / float(nsamp)) * random.uniform(0, 1)
    for j in range(nsamp):
        uj = u1 + j * (totalmass / float(nsamp))
        while uj > cws[i]:
            i += 1
        out[j] = i
    return out


##########################################################
def extract(region, mask):
    out = []
    for i in range(0, len(mask), 2):
        if (mask[i] == 255) and (region[i] > 0):
            out.append(i)
    print("found ", len(out), " test elements")
    return np.array(out)


##########################################################
def CNN2(num_classes, img_len, img_wdt, img_hgt, number_epochs, learning_rate):
    K.set_image_data_format("channels_first")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_len, img_hgt, img_wdt), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    decay = learning_rate / number_epochs
    sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model, number_epochs


##########################################################
def updatecat(catName, cat, map):
    if catName not in map:
        cat += 1
        map[catName] = cat
    else:
        cat = map[catName]
    print(cat, " -> ", map)
    return cat, map


##########################################################
EXPORT_SVG = True
number_epochs = 50
learning_rate = 0.01
images_directory = ""
D = None
train_size = None
valid_size = None
test_size = None
recompute = None
outdir = "identification_"

try:
    images_directory = sys.argv[1]
    D = int(sys.argv[2])
    train_size = int(sys.argv[3])
    valid_size = int(sys.argv[4])
    test_size = int(sys.argv[5])
    recompute = int(sys.argv[6])
except:
    print('Arguments: directory D train_size valid_size test_size recompute')
    quit()

if not os.path.isdir(images_directory):
    print(images_directory, " must be a directory.")
    quit()

list_images = load_data(images_directory)

computed_train = os.path.isfile(os.path.join(images_directory, "train.npz"))
computed_test = os.path.isfile(os.path.join(images_directory, "test.npz"))
computed_model = os.path.isfile(os.path.join(images_directory, "model.h5")) and os.path.isfile(os.path.join(images_directory, "model.json"))
computed_mapping = os.path.isfile(os.path.join(images_directory, "mapping.json"))

if computed_train and not recompute:
    train_data = np.load(os.path.join(images_directory, "train.npz"))
    x_train = train_data['x_train']
    y_train = train_data['y_train']
    x_valid = train_data['x_valid']
    y_valid = train_data['y_valid']

if computed_mapping and not recompute:
    with open(os.path.join(images_directory, "mapping.json"), 'r') as fp:
        mapping = json.load(fp)

if computed_test and not recompute:
    test_data = np.load(os.path.join(images_directory, "test.npz"))
    x_test = test_data['x_test']
    y_test = test_data['y_test']
    yI_test = test_data['yI_test']

if recompute:
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    mapping = {}
    x_test = []
    y_test = []
    yI_test = []

category = -1

for img in list_images:
    image_stack_path, annot_path, mask_train_path, mask_valid_path, mask_test_path = img
    if os.path.isfile(annot_path):
        image_stack = io.imread(image_stack_path)
        length, width, height = image_stack.shape
        annot2d = io.imread(annot_path)  # assuming annot has the same x-y dimensions as the original stack
        annot = annot2d.reshape(np.prod(annot2d.shape))
        del annot2d
        ########
        annot = annot / 255.0
        if np.amax(annot) <= 0:
            continue
        print("%s\tl=%s\tw=%s\th=%s" % (image_stack_path, length, width, height))
        imageName = os.path.basename(image_stack_path).split(".")[0]
        bacteriaName = os.path.basename(image_stack_path).split(".")[0].split("_")[0]

        if EXPORT_SVG:
            f = open(os.path.join(images_directory, imageName + ".svg"), "w+")
            f.write("<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"%s\" width=\"%s\">\n" % (height, width))

        if not computed_train or recompute:
            mask_train = None
            if os.path.isfile(mask_train_path):
                mask_train2d = io.imread(mask_train_path)
                mask_train = mask_train2d.reshape(np.prod(mask_train2d.shape))
                del mask_train2d

            ##########################################################################################################
            sample_idx = sample(annot, mask_train, train_size)

            category, mapping = updatecat(bacteriaName + "_Pos", category, mapping)

            for idx in sample_idx:
                x = idx % width
                y = idx // width
                if (D / 2 <= x < width - D / 2) and (D / 2 <= y < height - D / 2):
                    if EXPORT_SVG:
                        f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"red\" fill-opacity=\"0.4\"/>\n" % (x + .5, y + .5, D / 2))
                    x_train.append(image_stack[:, int(x - D / 2):int(x + D / 2), int(y - D / 2):int(y + D / 2)])
                    y_train.append(category)

            mask_valid = None
            if os.path.isfile(mask_valid_path):
                mask_valid2d = io.imread(mask_valid_path)
                mask_valid = mask_valid2d.reshape(np.prod(mask_valid2d.shape))
                del mask_valid2d

            ##########################################################################################################
            sample_idx = sample(annot, mask_valid, valid_size)

            category, mapping = updatecat(bacteriaName + "_Pos", category, mapping)

            for idx in sample_idx:
                x = idx % width
                y = idx // width
                if (D / 2 <= x < width - D / 2) and (D / 2 <= y < height - D / 2):
                    if EXPORT_SVG:
                        f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"red\" fill-opacity=\"0.2\"/>\n" % (x + .5, y + .5, D / 2))
                    x_valid.append(image_stack[:, int(x - D / 2):int(x + D / 2), int(y - D / 2):int(y + D / 2)])
                    y_valid.append(category)

        if not computed_test or recompute:
            mask_test = None
            if os.path.isfile(mask_test_path):
                mask_test2d = io.imread(mask_test_path)
                mask_test = mask_test2d.reshape(np.prod(mask_test2d.shape))
                del mask_test2d

            ##########################################################################################################
            sample_idx = sample(annot, mask_test, test_size)

            category, mapping = updatecat(bacteriaName + "_Pos", category, mapping)

            for idx in sample_idx:
                x = idx % width
                y = idx // width
                if (D / 2 <= x < width - D / 2) and (D / 2 <= y < height - D / 2):
                    if EXPORT_SVG:
                        f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"red\" fill-opacity=\"0.6\"/>\n" % (x + .5, y + .5, D / 2))
                    x_test.append(image_stack[:, int(x - D / 2):int(x + D / 2), int(y - D / 2):int(y + D / 2)])
                    y_test.append(category)
                    yI_test.append((imageName, x, y))

        if EXPORT_SVG:
            f.write("</svg>")
            f.close()

if recompute:
    np.savez(os.path.join(images_directory, "train.npz"), x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
    np.savez(os.path.join(images_directory, "test.npz"), x_test=x_test, y_test=y_test, yI_test=yI_test)
    with open(os.path.join(images_directory, "mapping.json"), 'w') as fp:
        json.dump(mapping, fp)

# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(x_train).astype('float32')
X_train = X_train / 255.0

X_valid = np.array(x_valid).astype('float32')
X_valid = X_valid / 255.0

# one hot encode outputs
y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train)
#
y_valid = np.array(y_valid)
y_valid = np_utils.to_categorical(y_valid)

X_test = np.array(x_test).astype('float32')
X_test = X_test / 255.0

y_test = np.array(y_test).astype('int')
np.savetxt(os.path.join(images_directory, 'y_test.txt'), y_test, fmt='%d')
y_test = np_utils.to_categorical(y_test)

print("Data normalized and hot encoded. Found ", len(mapping), " classes.")

print("CNN2 model... ")
model, epochs = CNN2(len(mapping), length, D, D, number_epochs, learning_rate)
print("created.")

# fit model
seed = 7
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=64)
print("model computed.")

# save the model
model.save_weights(os.path.join(images_directory, 'model_weights.h5'))
with open(os.path.join(images_directory, 'model_architecture.json'), 'w') as f:
    f.write(model.to_json())

y_pred = model.predict(X_test, batch_size=64)
y_pred = np.array(np.argmax(y_pred, axis=1)).astype('int')
np.savetxt(os.path.join(images_directory, 'y_pred.txt'), y_pred, fmt='%d')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
print("scores = ", scores)
print("Accuracy: %.2f%%" % (scores[1]*100))

f = open(os.path.join(images_directory, "params.txt"), "w+")
f.write("epoch_nr = %s\n" % (number_epochs))
f.write("l_rate = %s\n" % (learning_rate))
f.write("D = %s\n" % (D))
f.write("train_size = %s\n" % (train_size))
f.write("valid_size = %s\n" % (valid_size))
f.write("test_size = %s\n" % (test_size))
f.write("recompute = %s\n" % (recompute))
f.close()


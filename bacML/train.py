import os
from skimage import io
from random import random
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from b_tools import load_data, systematic_resampling, updatecat, grid_sampling, testfun
from b_models import mycnn
from timeit import default_timer as timer
import matplotlib.pyplot as plt

GRID_STEP = 2


def b2_train(train_directory="", patch_size=20, epoch_nr=10, learning_rate=0.001, method="mycnn", l2_reg=0.0):
    print('\n\ntrain_directory = {}\n'
          'patch_size={}\n'
          'epoch_nr={}\n'
          'learning_rate={}\n'
          'method={}\n'
          'l2_reg={}\n\n'.format(train_directory, patch_size, epoch_nr, learning_rate, method, l2_reg))

    if not os.path.isdir(train_directory):
        return
    if learning_rate < 0 or epoch_nr < 0 or patch_size < 0 or patch_size > 100:
        return
    if not (method in ['mycnn']):
        return

    train_directory_name = os.path.basename(train_directory)
    out_dir = os.path.join(train_directory, 'tr_mth_d_e_lr_l2r++%s_%s_%d_%d_%.9f_%.9f' % (train_directory_name, method, patch_size, epoch_nr, learning_rate, l2_reg))
    out_arch = os.path.join(out_dir, 'architecture.json')
    out_wgts = os.path.join(out_dir, 'weights.h5')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        if os.path.exists(out_arch):
            print("Found ", out_arch)
            return
        if os.path.exists(out_wgts):
            print("Found ", out_wgts)
            return

    list_images = load_data(train_directory, "tif", ("annot_pos", "annot_neg"))

    x_train = []
    y_train = []
    mapping = {}

    category = -1  # category counter

    for img in list_images:
        image_path, annot_pos_path, annot_neg_path = img
        image_name = os.path.basename(image_path).split(".")[0]
        bacteria_type = os.path.basename(image_path).split(".")[0].split("_")[0]

        annot_neg = io.imread(annot_neg_path) if os.path.isfile(annot_neg_path) else None
        annot_pos = io.imread(annot_pos_path) if os.path.isfile(annot_pos_path) else None

        if os.path.isfile(annot_pos_path) or os.path.isfile(annot_neg_path):
            image = io.imread(image_path)
            n_rows, n_cols = image.shape

            f = open(os.path.join(train_directory, image_name + ".svg"), "w+")
            f.write("<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"%s\" width=\"%s\">\n" % (n_rows, n_cols))

            if np.amax(annot_pos) > 0:
                category, mapping = updatecat(bacteria_type + "_pos", category, mapping)

                rows_pos, cols_pos = grid_sampling(annot_pos, GRID_STEP)

                for i in range(0, len(rows_pos)):
                    row = rows_pos[i]
                    col = cols_pos[i]

                    row0 = int(row - patch_size / 2)
                    row1 = int(row + patch_size / 2)

                    col0 = int(col - patch_size / 2)
                    col1 = int(col + patch_size / 2)

                    if row0 >= 0 and row1 < n_rows and col0 >= 0 and col1 < n_cols:
                        f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"red\" fill-opacity=\"0.1\"/>\n" % (col + .5, row + .5, .5))
                        x_train.append(image[row0:row1, col0:col1])
                        y_train.append(category)

                if np.amax(annot_neg) > 0:
                    category, mapping = updatecat(bacteria_type + "_neg", category, mapping)

                    annot_neg = annot_neg.reshape(np.prod(annot_neg.shape))
                    annot_neg = annot_neg / 255.0

                    for i in range(0, np.prod(annot_neg.shape)):
                        annot_neg[i] *= random()

                    sample_neg = systematic_resampling(annot_neg, None, len(rows_pos))

                    for i in range(0, len(sample_neg)):
                        row = sample_neg[i] // n_cols
                        col = sample_neg[i] % n_cols

                        row0 = int(row - patch_size / 2)
                        row1 = int(row + patch_size / 2)

                        col0 = int(col - patch_size / 2)
                        col1 = int(col + patch_size / 2)

                        if row0 >= 0 and row1 < n_rows and col0 >= 0 and col1 < n_cols:
                            f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"blue\" fill-opacity=\"0.1\"/>\n" % (col + .5, row + .5, .5))
                            x_train.append(image[row0:row1, col0:col1])
                            y_train.append(category)

            f.write("</svg>")
            f.close()

    # normalize inputs from 0-255 and 0.0-1.0
    X_train = np.array(x_train).astype('float32')
    X_train = X_train / 255.0

    # compute mean
    X_train_mean = np.mean(X_train, axis=(1, 2))
    # subtract mean
    for i in range(0, X_train.shape[0]):
        X_train[i] -= X_train_mean[i]

    X_train = np.reshape(X_train, X_train.shape + (1,))  # extend dimension
    # one hot encode outputs
    y_train = np.array(y_train)
    y_train = np_utils.to_categorical(y_train)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=7)

    if method == "mycnn":
        model = mycnn(X_train.shape[1:], len(mapping), l2_reg=l2_reg)
    else:
        return

    decay = learning_rate / epoch_nr
    # sgd = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay, nesterov=True)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.995, epsilon=1e-08, decay=decay, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    history_callback = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch_nr, batch_size=128)
    loss = np.array(history_callback.history["loss"])
    val_loss = np.array(history_callback.history["val_loss"])

    f = plt.figure()
    p1, = plt.plot(loss, color='blue')
    p2, = plt.plot(val_loss, color='red')
    plt.ylabel('loss')
    plt.legend((p1, p2), ('loss', 'validation loss'))
    f.savefig(os.path.join(out_dir, "loss.pdf"), bbox_inches='tight')

    model.save_weights(out_wgts)
    with open(out_arch, 'w') as f:
        f.write(model.to_json())


def b3_train():
    return None


mode = "quant"  # "ident"
t0 = timer()
for train_dir in ["C:\\Users\\10250153\\bacteria3\\train01"]:
    for patch_size in [16, 20, 24, 32]:
        for epoch_nr in [10]:
            for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                for method in ["mycnn"]:
                    for l2norm in [0.00, 0.01, 0.02]:
                        if mode == "quant":
                            b2_train(train_dir, patch_size, epoch_nr, learning_rate, method, l2norm)
                        elif mode == "ident":
                            # for this option there should be more bacteria classes in the train dir
                            b3_train()
t1 = timer()
print("Experiment took {0:.5f} sec.".format((t1 - t0)))

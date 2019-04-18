import sys
import os
import numpy as np
from skimage import io  # from skimage.io import imsave
from b_tools import get_locs
from keras.models import model_from_json
from timeit import default_timer as timer

POSITIVE_VALUE = 0
NEGATIVE_VALUE = 1  # two class problem
enforce_calculation = True
step = 1
circ_radius_ratio = 0.85

root = "C:\\Users\\10250153\\bacteria3"

tests = [
    ['test01', 'train01']#,
    # ['test02', 'train02']  # ,
    # ['test03', 'train03']
]

for t in tests:
    test_dir = os.path.join(root, t[0])  # where to look for the images and annotations
    if os.path.isdir(test_dir):
        print(test_dir)
        for dirname, dirnames, filenames in os.walk(os.path.join(root, t[1])):
            for filename in filenames:
                if filename == 'architecture.json':  # os.path.basename(os.path.join(dirname, filename)) == 'architecture.json':
                    model_architecture_path = os.path.join(dirname, 'architecture.json')
                    model_weights_path = os.path.join(dirname, 'weights.h5')
                    if not os.path.isfile(model_weights_path):
                        continue

                    # architecture
                    json_file = open(model_architecture_path, 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    model = model_from_json(loaded_model_json)
                    # weights
                    model.load_weights(model_weights_path)
                    model.summary()

                    model_info_str = os.path.basename(dirname)
                    model_info = model_info_str.split("++")
                    if not len(model_info) == 2:
                        continue
                    model_par = model_info[0].split("_")
                    model_parval = model_info[1].split("_")
                    print(model_par, "  --  ", model_parval)

                    for filename1 in os.listdir(os.path.join(root, t[0])):
                        if filename1.endswith(".tif"):
                            test_image = io.imread(os.path.join(root, t[0], filename1))
                            if len(test_image.shape) != 2:
                                continue
                            n_rows, n_cols = test_image.shape
                            test_image_name = filename1

                            test_image_annotation_path = os.path.join(root, t[0], "annot", test_image_name)
                            annotation_found = os.path.exists(test_image_annotation_path)
                            test_image_annotation = io.imread(test_image_annotation_path) if annotation_found else None

                            dmap = np.ndarray(shape=(n_rows, n_cols), dtype='uint8') * 0
                            dmap_dir = os.path.join(root, t[0], model_info_str)
                            if not os.path.exists(dmap_dir):
                                os.makedirs(dmap_dir)

                            previous_detection = None
                            previous_detection_path = os.path.join(dmap_dir, test_image_name + ".npz")
                            if not enforce_calculation and os.path.isfile(previous_detection_path):
                                previous_detection = np.load(previous_detection_path)  # y_pred, r, c
                                y_pred = previous_detection['y_pred']
                                r = previous_detection['r']
                                c = previous_detection['c']
                                y_test = [None] * len(r) if annotation_found else None
                                for i in range(0, len(r)):
                                    if annotation_found:
                                        y_test[i] = POSITIVE_VALUE if test_image_annotation[
                                                                          r[i], c[i]] == 255 else NEGATIVE_VALUE  # to be compliant with b3_train and the class index formation with 2 classes
                            else:
                                D_rows = int(model.layers[0].input.shape[1])
                                D_cols = int(model.layers[0].input.shape[2])

                                r, c = get_locs(n_rows, n_cols, step, D_rows, D_cols, circ_radius_ratio)

                                cnt = 0
                                x_test = [None] * len(r)
                                y_test = [None] * len(r) if annotation_found else None

                                for i in range(0, len(r)):
                                    row0 = int(r[i] - D_rows / 2)
                                    row1 = int(r[i] + D_rows / 2)
                                    col0 = int(c[i] - D_cols / 2)
                                    col1 = int(c[i] + D_cols / 2)
                                    x_test[i] = test_image[row0:row1, col0:col1]
                                    if annotation_found:
                                        y_test[i] = POSITIVE_VALUE if test_image_annotation[
                                                                          r[i], c[i]] == 255 else NEGATIVE_VALUE  # to be compliant with b3_train and the class index formation with 2 classes

                                # compute X_test
                                X_test = np.array(x_test).astype('float32')
                                X_test = X_test / 255.0

                                # compute mean
                                X_test_mean = np.mean(X_test, axis=(1, 2))
                                # subtract mean
                                for i in range(0, X_test.shape[0]):
                                    X_test[i] -= X_test_mean[i]

                                X_test = np.reshape(X_test, X_test.shape + (1,))

                                # compute y_pred
                                print("Computing y_pred... ")
                                t0 = timer()
                                y_pred = model.predict(X_test, batch_size=128)
                                y_pred = np.array(np.argmax(y_pred, axis=1)).astype('int')
                                t1 = timer()
                                print("done")
                                print(" {0:.5f} sec.".format((t1 - t0)))
                                np.savez(previous_detection_path, y_pred=y_pred, r=r, c=c)

                                # fill in the detection map, or fill in the svg with the detections
                                for i in range(0, len(y_pred)):
                                    dmap[r[i], c[i]] = 255 if y_pred[i] == POSITIVE_VALUE else 0  # depends on how the classes are indexed in train

                                io.imsave(os.path.join(dmap_dir, test_image_name), dmap)
                                # io.imshow(dmap)
                                # plt.show()

                            if annotation_found:
                                tp, fp, fn, precision, recall, f1 = (0, 0, 0, 0, 0, 0)
                                for i in range(0, len(y_pred)):
                                    if y_pred[i] == POSITIVE_VALUE and y_test[i] == POSITIVE_VALUE:
                                        tp += 1
                                    elif y_pred[i] == POSITIVE_VALUE and y_test[i] == NEGATIVE_VALUE:
                                        fp += 1
                                    elif y_pred[i] == NEGATIVE_VALUE and y_test[i] == POSITIVE_VALUE:
                                        fn += 1
                                precision = (tp / (tp + fp)) if (tp + fp > 0) else 0
                                recall = (tp / (tp + fn)) if (tp + fn > 0) else 0
                                f1 = (2 * precision * recall) / (precision + recall)

                                f = open(os.path.join(root, t[0], "eval.csv"), "a")  # "w+"
                                # legend
                                # imname, tp, fp, fn, p, r, f1, tr, mth, D, e, learning_rate, l2r
                                f.write("%s, %d, %d, %d, %f, %f, %f, %s, %s, %d, %d, %f, %f\n" % (
                                    test_image_name, tp, fp, fn, precision, recall, f1,
                                    model_parval[0],
                                    model_parval[1],
                                    int(model_parval[2]),
                                    int(model_parval[3]),
                                    float(model_parval[4]),
                                    float(model_parval[5])))
                                f.close()

model_weights_path = os.path.join(os.path.abspath(os.path.join(model_architecture_path, os.pardir)), "weights.h5")
model_info_str = os.path.basename(os.path.abspath(os.path.join(model_architecture_path, os.pardir)))
model_info = model_info_str.split("++")

print(model_info)

model_par = model_info[1].split("_")
model_parval = model_info[2].split("_")

model_info = {
    "name": model_info[0],
    model_par[0]: int(model_parval[0]),
    model_par[1]: int(model_parval[1]),
    model_par[2]: float(model_parval[2]),
    model_par[3]: float(model_parval[3])
}

print(model_info)

if (
        os.path.isfile(test_image_path) and test_image_path.endswith(".tif") and
        os.path.isfile(model_architecture_path) and model_architecture_path.endswith(".json") and
        os.path.isfile(model_weights_path) and model_weights_path.endswith(".h5")
):
    test_image = io.imread(test_image_path)

    if len(test_image.shape) > 2:
        print("stopping... test image needs to be 2d")
        quit()

    n_rows, n_cols = test_image.shape

    test_image_name = os.path.basename(test_image_path)

    # test_image_annot_name = os.path.basename(test_image_path).split(".")[0].split("_")[0]

    test_image_parent_dir = os.path.abspath(os.path.join(test_image_path, os.pardir))

    test_image_annotation_path = os.path.join(test_image_parent_dir, "annot", test_image_name)
    annotation_found = os.path.exists(test_image_annotation_path)
    test_image_annotation = io.imread(test_image_annotation_path) if annotation_found else None

    # architecture
    json_file = open(model_architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # weights
    model.load_weights(model_weights_path)
    # model.summary()

    ####
    dmap = np.ndarray(shape=(n_rows, n_cols), dtype='uint8') * 0
    dmap_dir = os.path.join(test_image_parent_dir, 'det++' + model_info_str)
    if not os.path.exists(dmap_dir):
        os.makedirs(dmap_dir)

    ####
    previous_detection = None
    if not enforce_calculation and os.path.isfile(os.path.join(dmap_dir, test_image_name + ".npz")):
        previous_detection = np.load(os.path.join(dmap_dir, test_image_name + ".npz"))  # y_pred, r, c
        y_pred = previous_detection['y_pred']
        r = previous_detection['r']
        c = previous_detection['c']
        y_test = [None] * len(r) if annotation_found else None
        for i in range(0, len(r)):
            if annotation_found:
                y_test[i] = 0 if test_image_annotation[r[i], c[i]] == 255 else 1  # to be compliant with b3_train and the class index formation with 2 classes
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
                y_test[i] = 0 if test_image_annotation[r[i], c[i]] == 255 else 1  # to be compliant with b3_train and the class index formation with 2 classes

        # compute X_test
        X_test = np.array(x_test).astype('float32')
        X_test = X_test / 255.0  # TODO: subtract mean
        X_test = np.reshape(X_test, X_test.shape + (1,))

        # compute y_pred
        y_pred = model.predict(X_test, batch_size=128)
        y_pred = np.array(np.argmax(y_pred, axis=1)).astype('int')

        np.savez(os.path.join(dmap_dir, test_image_name + ".npz"), y_pred=y_pred, r=r, c=c)

        # fill in the detection map, or fill in the svg with the detections
        for i in range(0, len(y_pred)):
            dmap[r[i], c[i]] = 255 if y_pred[i] == 0 else 0  # depends on how the classes are indexed in train

        io.imsave(os.path.join(dmap_dir, test_image_name), dmap)
        # io.imshow(dmap)
        # plt.show()

    tp, fp, fn, precision, recall, f1 = (0, 0, 0, 0, 0, 0)

    if annotation_found:
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

    else:
        tp, fp, fn, precision, recall, f1 = (-99999, -99999, -99999, -99999, -99999, -99999)

    f = open(os.path.join(dmap_dir, "eval.txt"), "a")  # "w+"
    f.write("imname, tp, fp, fn, p, r, f1, method, D, epch, learning_rate, l2r\n")
    f.write("%s, %d, %d, %d, %f, %f, %f, %s, %d, %d, %f, %f\n" % (
        test_image_name, tp, fp, fn, precision, recall, f1,
        model_info['name'],
        model_info['D'],
        model_info['epch'],
        model_info['learning_rate'],
        model_info['l2r']))  # model_info[model_par[3]]
    f.close()

print("done.")

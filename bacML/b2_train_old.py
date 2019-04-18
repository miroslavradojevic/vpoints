##############################################
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

        ##########################################################
        if np.amax(annot_pos) > 0:
            category, mapping = updatecat(bacteria_type + "_pos", category, mapping)
            rows_pos, cols_pos = grid_sampling(annot_pos, 2)
            for i in range(0, len(rows_pos)):
                row = rows_pos[i]
                col = cols_pos[i]

                row0 = int(row - D / 2)
                row1 = int(row + D / 2)

                col0 = int(col - D / 2)
                col1 = int(col + D / 2)

                if row0 >= 0 and row1 < n_rows and col0 >= 0 and col1 < n_cols:
                    f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"red\" fill-opacity=\"0.1\"/>\n" % (col + .5, row + .5, D / 2))
                    x_train.append(image[row0:row1, col0:col1])
                    y_train.append(category)

            print("Positives = ", len(rows_pos))

            ##########################################################
            if np.amax(annot_neg) > 0:
                category, mapping = updatecat(bacteria_type + "_neg", category, mapping)
                annot_neg = annot_neg.reshape(np.prod(annot_neg.shape))
                annot_neg = annot_neg / 255.0
                for ii in range(0, np.prod(annot_neg.shape)):
                    annot_neg[ii] *= random()
                sample_neg = systematic_resampling(annot_neg, None, len(rows_pos))
                print("Negatives = ", len(sample_neg))
                for i in range(0, len(sample_neg)):
                    row = sample_neg[i] // n_cols
                    col = sample_neg[i] % n_cols

                    row0 = int(row - D / 2)
                    row1 = int(row + D / 2)

                    col0 = int(col - D / 2)
                    col1 = int(col + D / 2)

                    if row0 >= 0 and row1 < n_rows and col0 >= 0 and col1 < n_cols:
                        f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"blue\" fill-opacity=\"0.1\"/>\n" % (col + .5, row + .5, D / 2))
                        x_train.append(image[row0:row1, col0:col1])
                        y_train.append(category)

        f.write("</svg>")
        f.close()

print("Train set size = ", len(x_train))

# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(x_train).astype('float32')
X_train = X_train / 255.0  # TODO: subtract mean
X_train_mean = X_train.mean(axis = 1, keepdims=True)
print(X_train.shape)
print(X_train_mean.shape)
if True:
    quit()
X_train = np.reshape(X_train, X_train.shape + (1,))  # extend dimension
# one hot encode outputs
y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=7)

if method == "model01":
    model = model01(len(mapping), X_train.shape[1:], epoch_nr, l_rate)
elif method == "mycnn":
    model = alexnet_model((D, D, 1), len(mapping), l2_reg=l2_reg)

# Compile model
decay = l_rate / epoch_nr
# sgd = SGD(learning_rate=l_rate, momentum=0.9, decay=decay, nesterov=True)
adam = Adam(lr=l_rate, beta_1=0.9, beta_2=0.995, epsilon=1e-08, decay=decay, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

history_callback = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch_nr, batch_size=64)

loss = np.array(history_callback.history["loss"])
val_loss = np.array(history_callback.history["val_loss"])

f = plt.figure()
p1, = plt.plot(loss, color='blue')
p2, = plt.plot(val_loss, color='red')
plt.ylabel('loss')
plt.legend((p1, p2), ('loss', 'validation loss'))
f.savefig(os.path.join(out_dir, "loss.pdf"), bbox_inches='tight')

np.savetxt(os.path.join(out_dir, "loss.txt"), loss, delimiter=",")
np.savetxt(os.path.join(out_dir, "val_loss.txt"), val_loss, delimiter=",")

model.save_weights(os.path.join(out_dir, 'weights.h5'))
with open(os.path.join(out_dir, 'architecture.json'), 'w') as f:
    f.write(model.to_json())





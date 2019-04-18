import os
import sys
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



width = 2000
height = 2000
D = None

try:
    directory = sys.argv[1]
    D = int(sys.argv[2])
except:
    quit()

if not os.path.isdir(directory):
    quit()

test_data = np.load(os.path.join(directory, "test.npz"))

y_test = np.array(np.loadtxt(os.path.join(directory, "y_test.txt"))).astype('int')
y_pred = np.array(np.loadtxt(os.path.join(directory, "y_pred.txt"))).astype('int')
with open(os.path.join(directory, "mapping.json"), 'r') as fp:
    mapping = json.load(fp)
    print(mapping)
    print(len(mapping))



class_names = []
for i in range(len(mapping)):
    class_names.append("")
print(class_names)
for tt in mapping:
    class_names[mapping[tt]] = tt
print(class_names)

print("len y_pred = ", len(y_pred))
print("len y_test = ", len(y_test))

tp = 0
fp = 0
fn = 0

for i in range(len(y_pred)):
    if y_pred[i] == 1 and y_test[i] == 1:
        tp += 1
    elif y_pred[i] == 1 and y_test[i] == 0:
        fp += 1
    elif y_pred[i] == 0 and y_test[i] == 1:
        fn += 1
print("tp=", tp, ", fp=", fp, ", fn=", fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1 = ((2*precision*recall)/(precision+recall)) if (precision + recall) > 0 else 0
print("p=", precision,", r=", recall, ", f=", f1)

f = open(os.path.join(directory, "eval.txt"), "w+")
f.write("tp fp fn p r f1\n" )
f.write("%s %s %s %s %s %s\n" % (tp, fp, fn, precision, recall, f1))
f.close()

yI_test = np.array(test_data['yI_test'])
y_test = np.array(test_data['y_test'])

imNamesArray = yI_test[:, 0]
x_coord = yI_test[:, 1].astype('float32')
y_coord = yI_test[:, 2].astype('float32')

imNamesSet = set(imNamesArray)
print(len(imNamesArray))

for imName in imNamesSet:
    print(imName)
    f = open(os.path.join(directory, imName + "_det.svg"), "w+")
    f.write("<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"%s\" width=\"%s\">\n" % (height, width))
    for i in range(len(imNamesArray)):
        if imNamesArray[i] == imName:
            if y_pred[i] == 1:
                col = "red"
            else:
                col = "blue"
            f.write("<circle cx=\"%s\" cy=\"%s\" r=\"%s\" fill=\"%s\" fill-opacity=\"0.4\"/>\n" % (x_coord[i] + .5, y_coord[i] + .5, D / 2, col))
    f.write("</svg>")
    f.close()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
# plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

print("done")




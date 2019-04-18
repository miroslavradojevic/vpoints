import os
import sys

try:
    imagepath = sys.argv[1]
    # D = int(sys.argv[2])
    # train_size = int(sys.argv[3])
    # valid_size = int(sys.argv[4])
    # test_size = int(sys.argv[5])
    # recompute = int(sys.argv[6])
except:
    print('Usage:\n python vpdetector.py p1 p2')
    quit()

if not os.path.isfile(imagepath):
    print(imagepath, " must be a file")
    quit()

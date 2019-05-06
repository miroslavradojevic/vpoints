import numpy as np
from scipy.signal import argrelextrema

a = np.array([3,2,3,4,5,5,4,3,2,1,2,3,2,1,2,3,4,5,6,5,4,3,2,3])

# determine the indices of the local maxima
maxInd = argrelextrema(a, np.greater)

# get the actual values using these indices
r = a[maxInd]  # array([5, 3, 6])
print("r=", r, " idx=", maxInd)

from pylab import *
plot(a)
show()
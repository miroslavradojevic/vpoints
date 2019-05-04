import math
import numpy
import matplotlib.pyplot as plt

y = []

x_r = numpy.arange(-1.0, 1.0, 0.02)

for x in x_r:
    print(math.acos(x));
    y.extend([math.acos(x)]);

print(len(y), " ", len(x_r), " ", )

plt.plot(x_r, y, 'ro')
plt.show()
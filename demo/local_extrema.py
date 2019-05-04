from numpy import *

# example data with some peaks:
x = linspace(0,4,1e1)
data = .2*sin(10*x)+ exp(-abs(2-x)**2)

# that's the line, you need:
a = diff(sign(diff(data))).nonzero()[0] + 1 # local min+max
b = (diff(sign(diff(data))) > 0).nonzero()[0] + 1 # local min
c = (diff(sign(diff(data))) < 0).nonzero()[0] + 1 # local max


# graphical output...
from pylab import *
plot(x,data)
plot(x[b], data[b], "o", label="min")
plot(x[c], data[c], "o", label="max")
legend()
show()
import datetime

t0 = datetime.datetime.now()

N = []
for i in range(1200000):
    N.append([])

t1 = datetime.datetime.now()
dt = t1 - t0
print( (dt.microseconds / 1000), "ms")

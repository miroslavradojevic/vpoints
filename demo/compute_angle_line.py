import math

p1x = 10
p1y = 10

p2 = [[0, 0], [0, 10], [0, 20], [10, 0], [10, 20], [20, 0], [20, 10], [20, 20]]

for i in range(len(p2)):
    angle = math.atan2(p2[i][1] - p1y, p2[i][0] - p1x)
    print("x = ", p2[i][0], "y = ", p2[i][1], angle, "rad", (angle * 180.0 / math.pi), "deg")

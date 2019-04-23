import math


def cross(v1x, v1y, v2x, v2y):
    return v1x * v2y - v1y * v2x


def modulus(vx, vy):
    return math.sqrt(vx * vx + vy * vy)
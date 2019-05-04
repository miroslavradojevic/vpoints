import math


def cross(v1x, v1y, v2x, v2y):
    return v1x * v2y - v1y * v2x


def modulus(vx, vy):
    return math.sqrt(vx * vx + vy * vy)


def dot(v1x, v1y, v2x, v2y):
    return v1x * v2x + v1y * v2y


def direction(px, py, rx, ry):
    l = math.sqrt(math.pow(rx - px, 2) + math.pow(ry - py, 2))
    return (rx - px) / l, (ry - py) / l


def weight(x, y, px, py, vpx, vpy):
    ll = math.sqrt(math.pow(x - px, 2) + math.pow(y - py, 2))
    if ll > 0:
        dp = dot((x - px) / ll, (y - py) / ll, vpx, vpy)
        if dp > 0:
            return dp
        else:
            return 0
    else:
        return 0

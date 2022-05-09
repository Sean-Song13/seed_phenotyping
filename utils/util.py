from collections import namedtuple
import math
import numpy as np
from skimage.draw import line, line_aa

Point = namedtuple('Point', ['x', 'y'])


class Axis(object):
    def __init__(self, angle, radius=None, score=None):
        self.angle = angle % 180
        self.radius = radius
        self.score = score

        radians = math.radians(angle)
        self.sin = math.sin(radians)
        self.cos = math.cos(radians)

    def __repr__(self):
        return str(self.angle)

    def __add__(self, angle):
        return Axis(angle=(self.angle + angle) % 180)


def pad_for_rotation(image, center=None):
    """ pad image to allow for rotation """
    h, w = image.shape[:2]

    if center is None:
        center = Point(w // 2, h // 2)

    max_x = max(center.x, w - center.x)
    max_y = max(center.y, h - center.y)
    # r = int(math.sqrt(max_x**2 + max_y**2))
    r = max(max_x, max_y)

    padding = Point(int(r - center.x), int(r - center.y))
    padded_image = np.zeros((r * 2, r * 2) + image.shape[2:], dtype=image.dtype)
    padded_image[..., :3] = 0
    padded_image[padding.y:padding.y + h, padding.x:padding.x + w] = image

    return padded_image, center


def draw_axes(image, center, *axes):
    """ overlay image with axes """
    out = image.copy()
    size = max(out.shape[:2])
    for axis in axes:
        dx = size * axis.cos
        dy = size * -axis.sin

        bias_x = 0
        bias_y = 0
        if 45 < axis.angle < 135:
            bias_x = round(axis.radius / axis.sin)
        else:
            bias_y = round(axis.radius / axis.cos)

        x0, y0 = int(round(center.x - dx) - bias_x), int(round(center.y - dy) - bias_y)
        x1, y1 = int(round(center.x + dx) - bias_x), int(round(center.y + dy) - bias_y)

        coords = line_aa(y0, x0, y1, x1)
        for r, c, val in zip(*coords):
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = val * 255
    return out


def get_axis(size, center, axis):
    dx = size * axis.cos
    dy = size * -axis.sin

    bias_x = 0
    bias_y = 0
    if 45 < axis.angle < 135:
        bias_x = round(axis.radius / axis.sin)
    else:
        bias_y = round(axis.radius / axis.cos)

    x0, y0 = int(round(center.x - dx) - bias_x), int(round(center.y - dy) - bias_y)
    x1, y1 = int(round(center.x + dx) - bias_x), int(round(center.y + dy) - bias_y)

    r, c = line(y0, x0, y1, x1)
    if 45 < axis.angle < 135:
        start = c[np.where(r == 0)[0][0]]
        end = c[np.where(r == center[1] * 2 - 1)[0][0]]
        return [start, 0], [end, center[1] * 2]
    else:
        start = r[np.where(c == 0)[0][0]]
        end = r[np.where(c == size - 1)[0][0]]
        return [0, start], [size, end]


def orient(p, q, r):
    M = np.array([[1, p[0], p[1]], [1, q[0], q[1]], [1, r[0], r[1]]]).transpose()
    det = np.linalg.det(M)
    return det

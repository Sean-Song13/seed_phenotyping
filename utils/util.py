#
# Gemini -- utility functions
#
# Copyright (c) 2014-2015 Carnegie Mellon University
# All rights reserved.
#
# This software is distributed under the terms of the Eclipse Public
# License, Version 1.0 which can be found in the file named LICENSE.
# ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS SOFTWARE CONSTITUTES
# RECIPIENT'S ACCEPTANCE OF THIS AGREEMENT
#

from collections import namedtuple

try:
    from itertools import izip
except ImportError:  # will be 3.x series
    pass
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.color import rgb2gray
from skimage.data import imread
from skimage.draw import line, line_aa
from skimage.measure import regionprops, label
from skimage.morphology import binary_opening, disk
from skimage.transform import rescale

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


def read_image(name, size=None, debug=False):
    """ read image and segmentation, returns RGB + alpha composite """
    image = imread(name) / 255.

    if image.shape[2] == 4:
        alpha = image[..., 3]
        image = image[..., :3]
    else:
        # segmentation_name = os.path.splitext(name)[0][:-6] + '-label.png'
        segmentation = image
        alpha = np.ones_like(segmentation[..., 0])
        # plt.imshow(alpha)
        # plt.title("alpha")
        # plt.show()

    if size is not None:
        scale_x = float(size[0]) / image.shape[1]
        scale_y = float(size[1]) / image.shape[0]
        scale = min(scale_x, scale_y)

        if debug:
            print(name, size[0], size[1], image.shape[1], image.shape[0], scale, image.shape[1] * scale,
                  image.shape[0] * scale)

        if scale > 1.0:
            print('Image %s smaller than requested size' % name)

        if scale != 1.0:
            image = rescale(image, scale, order=3)
            alpha = rescale(alpha, scale, order=0)

    return np.dstack((image, alpha))


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


def crop_lesion(image, attrs={}, debug=False):
    """ crop image to the segmented lesion """
    cleaned = binary_opening(image[..., 3], disk(3))
    labels = label(cleaned, 4, background=0)
    props = regionprops(labels + 1, intensity_image=rgb2gray(image))

    sizes = [(p['area'], i) for i, p in enumerate(props)]
    max_value, max_index = max(sizes)
    p = props[max_index]

    # plt.imshow(image[..., 3])
    # plt.show()
    bbox = p['bbox']
    image_roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :3].copy()
    # plt.imshow(image_roi)
    # plt.title()
    # plt.show()
    # alpha_roi = p['filled_image'] # or 'image'
    alpha_roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3], 3].copy()

    if debug:
        print("""\
Image properties
----------------
Area        : %d    (%d filled)
Centroid    : (%.3f, %.3f)
Axes        : (%.3f, %.3f)
Orientation : %.3f
""" % (
            p['area'], p['filled_area'],
            p['centroid'][1], p['centroid'][0],
            p['major_axis_length'], p['minor_axis_length'],
            math.degrees(p['orientation'])
        ))
        plt.subplot(221)
        plt.imshow(image[..., 3])
        plt.title("rgb")
        plt.subplot(222)
        plt.imshow(labels)
        plt.subplot(223)
        imgplot = plt.imshow(image_roi)
        plt.title("image_roi")
        imgplot.set_interpolation('nearest')
        plt.subplot(224)
        imgplot = plt.imshow(alpha_roi)
        plt.title("alpha_roi")
        imgplot.set_interpolation('nearest')
        plt.show()

    attrs.update([
        ('Lesion Area', int(p['area'])),
        ('Lesion Area Filled', p['filled_area']),
        ('--Lesion Centroid X', p['centroid'][1]),
        ('--Lesion Centroid Y', p['centroid'][0]),
        ('--Lesion Orientation', math.degrees(p['orientation']) % 180),
        ('--Lesion Extent', p['extent']),
        ('Lesion Perimeter', p['perimeter']),
        ('Lesion Solidity', p['solidity']),
        ('Lesion Major Axis Length', p['major_axis_length']),
        ('Lesion Minor Axis Length', p['minor_axis_length']),
        ('--Lesion Bounding Box X1', p['bbox'][1]),
        ('--Lesion Bounding Box Y1', p['bbox'][0]),
        ('--Lesion Bounding Box X2', p['bbox'][3]),
        ('--Lesion Bounding Box Y2', p['bbox'][2]),
        ('Lesion Intensity Min', p['min_intensity']),
        ('Lesion Intensity Mean', p['mean_intensity']),
        ('Lesion Intensity Max', p['max_intensity']),
        ('Lesion Eccentricity', p['eccentricity']),
        ('Lesion FormFactor', p['area'] / (p['perimeter'] ** 2)),
    ])
    return np.dstack((image_roi, alpha_roi))


def scale_lesion(lesion, size):
    """ scale segmented lesion to uniform size """
    image = lesion[..., :3]
    alpha = lesion[..., 3]

    scale = float(size) / max(*alpha.shape)
    if scale != 1.0:
        image = rescale(image, scale, order=1)
        alpha = rescale(alpha, scale, order=1)

    return np.dstack((image, alpha))


def image_roi_offset(attrs, shape):
    """ extract bbox/scale from attrs """
    h, w = shape[:2]
    x1 = attrs.get('--Lesion Bounding Box X1', 0)
    y1 = attrs.get('--Lesion Bounding Box Y1', 0)
    x2 = attrs.get('--Lesion Bounding Box X2', w)
    y2 = attrs.get('--Lesion Bounding Box Y2', h)
    scale = float(max(w, h)) / max(x2 - x1, y2 - y1)
    return (y1, x1, y2, x2), scale


def sample_pixels(image, N, mask=None):
    """ Randomly sample pixels from the image within the masked region """
    if mask is None:
        h, w = image.shape[:2]
        samples = np.random.choice(h * w, N)
        y, x = samples // w, samples % w
        return image[y, x]
    else:
        y, x = np.where(mask)
        samples = np.random.choice(len(y), N)
        return image[y[samples], x[samples]]


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
        end = c[np.where(r == center[1]*2 - 1)[0][0]]
        return [start, 0], [end, center[1]*2]
    else:
        start = r[np.where(c == 0)[0][0]]
        end = r[np.where(c == size - 1)[0][0]]
        return [0, start], [size, end]


def segmentation_edges(image):
    """ number of edges touched by image segmentation """
    lesion_mask = image[..., 3]
    edges = sum([
        lesion_mask[0].any(),
        lesion_mask[-1].any(),
        lesion_mask[:, 0].any(),
        lesion_mask[:, -1].any(),
    ])
    return edges


def dis_p2line(x, start, end):
    line_vec = end - start
    p_vec = x - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    p_vec_scaled = 1.0 / line_len * p_vec
    inner_product = np.dot(line_unit, p_vec_scaled)
    inner_product = np.clip(inner_product, 0.0, 1.0)
    nearest = start + line_vec * inner_product
    distance = np.linalg.norm(x - nearest)
    return distance


def orient(p, q, r):
    M = np.array([[1, p[0], p[1]], [1, q[0], q[1]], [1, r[0], r[1]]]).transpose()
    det = np.linalg.det(M)
    return det

#
# Gemini -- Planar-Reflective Symmetry Transform
#
# Copyright (c) 2014-2015 Carnegie Mellon University
# All rights reserved.
#
# This software is distributed under the terms of the Eclipse Public
# License, Version 1.0 which can be found in the file named LICENSE.
# ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS SOFTWARE CONSTITUTES
# RECIPIENT'S ACCEPTANCE OF THIS AGREEMENT
#
# @reference
#  Joshua Podolak, Philip Shilane, Aleksey Golovinskiy, Szymon Rusinkiewicz,
#  and Thomas Funkhouser. A Planar-Reflective Symmetry Transform for 3D Shapes.
#  ACM Transactions on Graphics (Proc. SIGGRAPH). 25(3) July 2006
#
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from skimage.transform import rotate

from utils.util import Point, Axis, pad_for_rotation, image_roi_offset, draw_axes


def _symToXY(axis, image_center=None):
    """ convert from symmetrymap coordinates to roi coordinates """
    if image_center is None: image_center = Point(0, 0)

    X = (axis.radius * axis.sin) / 2 + image_center.x
    Y = (axis.radius * -axis.cos) / 2 + image_center.y
    return Point(X, Y)


def _find_intersection(axis1, axis2, image_center=None):
    """ estimate symmetry center from the major and minor axes.
    takes into account that the image's vertical axis points down
    """
    if image_center is None: image_center = Point(0, 0)

    dt = axis1.cos * axis2.sin - axis1.sin * axis2.cos
    assert (dt != 0)  # axes must not be parallel!

    x = (axis2.cos * axis1.radius - axis1.cos * axis2.radius) / dt + image_center.x
    y = (axis1.sin * axis2.radius - axis2.sin * axis1.radius) / dt + image_center.y
    return Point(x, y)


def compute_upper_map(start, end, imgR, imgF, symmetryMap, angle):
    size = imgR.shape[0]
    for offset in range(start, end):
        idxOffset = offset - start
        if size < 350:
            expect = imgR[:offset].ravel().dot(imgR[:offset].ravel())
            reflect = imgR[:offset].ravel().dot(imgF[(size - offset):].ravel())
        else:
            expect = np.einsum('ij,ij', imgR[:offset], imgR[:offset])
            reflect = np.einsum('ij,ij', imgR[:offset], imgF[(size - offset):])
        symmetryMap[angle, idxOffset] = 0.0 if expect == 0 else reflect / expect


def compute_lower_map(start, end, imgR, imgF, symmetryMap, angle, begin_offset):
    size = imgR.shape[0]
    for offset in range(start, end):
        idxOffset = offset - begin_offset
        offset -= size
        if size < 350:
            expect = imgR[offset:].ravel().dot(imgR[offset:].ravel())
            reflect = imgR[offset:].ravel().dot(imgF[:(size - offset)].ravel())
        else:
            expect = np.einsum('ij,ij', imgR[offset:], imgR[offset:])
            reflect = np.einsum('ij,ij', imgR[offset:], imgF[:(size - offset)])
        symmetryMap[angle, idxOffset] = 0.0 if expect == 0 else reflect / expect


def PlanarReflectiveSymmetryTransform(image_roi, direction="h", anglestep=1, debug=False):
    """ Planar-Reflective Symmetry Transform """
    start_timer = time.perf_counter()
    roi, image_center = pad_for_rotation(image_roi[..., 0])
    size = roi.shape[0]
    raw_anglestep = 4
    anglestep = raw_anglestep
    ############################################
    # apply Planar-Reflective Symmetry Transform
    angles, remainder = divmod(180, anglestep)
    assert (remainder == 0)

    num_offsets = 2 * size - 1
    symmetryMap = np.zeros((angles, num_offsets))

    # for each angle, rotate the image, and compare with reflection
    for angle in range(0, angles):
        imgR = rotate(roi, -angle * anglestep)
        imgF = imgR[::-1]

        compute_upper_map(0, size, imgR, imgF, symmetryMap, angle)
        compute_lower_map(size, num_offsets, imgR, imgF, symmetryMap, angle, 0)

    ############################################
    if debug:
        plt.imshow(symmetryMap)
        plt.show()
    anglestep = 180 / symmetryMap.shape[0]
    # find axes by looking for local maxima
    symScore = np.max(symmetryMap)
    maxima = peak_local_max(symmetryMap, threshold_abs=symScore * .5, threshold_rel=.6,
                            min_distance=5, num_peaks=5000, exclude_border=False)
    if direction == "v":
        maxima = np.array([m if 45 < m[0] * anglestep < 135 else np.array([-1, -1]) for m in maxima])
    else:
        maxima = np.array(
            [m if 45 > m[0] * anglestep or m[0] * anglestep > 135 else np.array([-1, -1]) for m in maxima])
    peak_scores = symmetryMap[maxima.T[0], maxima.T[1]]
    order = np.argsort(peak_scores)[::-1]
    idxMajor = order[0]
    angle, raw_offset = maxima[idxMajor]
    raw_angle = angle * anglestep

    angle_range = raw_anglestep
    offset_range = size // 5
    num_offsets = offset_range * 2 - 1
    fine_anglestep = 0.1
    angles = int(angle_range * 2 / fine_anglestep) + 1
    symmetryMap = np.zeros((angles, num_offsets))

    for angle in np.arange(raw_angle - angle_range, raw_angle + angle_range + fine_anglestep, fine_anglestep):
        imgR = rotate(roi, -angle)
        imgF = imgR[::-1]
        idxAngle = int(round((angle + angle_range - raw_angle) / fine_anglestep, 1))

        if raw_offset - offset_range < size < raw_offset + offset_range:
            compute_upper_map(raw_offset - offset_range, size, imgR, imgF, symmetryMap, idxAngle)
            compute_lower_map(size, raw_offset + offset_range - 1, imgR, imgF, symmetryMap, idxAngle,
                              raw_offset - offset_range)

        elif raw_offset + offset_range < size:
            compute_upper_map(raw_offset - offset_range, raw_offset + offset_range - 1, imgR, imgF, symmetryMap,
                              idxAngle)

        else:
            compute_lower_map(raw_offset - offset_range, raw_offset + offset_range - 1, imgR, imgF, symmetryMap,
                              idxAngle,
                              raw_offset - offset_range)

    ############################################
    # find major and minor symmetry axes in symmetry map

    # find axes by looking for local maxima
    symScore = np.max(symmetryMap)
    maxima = peak_local_max(symmetryMap, threshold_abs=symScore * .2, threshold_rel=.6,
                            min_distance=20, num_peaks=5000, exclude_border=False)

    peak_scores = symmetryMap[maxima.T[0], maxima.T[1]]
    order = np.argsort(peak_scores)[::-1]

    idxMajor = order[0]
    angle, offset = maxima[idxMajor]
    angle = angle * fine_anglestep - angle_range + raw_angle
    offset += raw_offset - offset_range
    major_axis = Axis(
        angle=angle,
        radius=(size - offset) / 2,
        score=peak_scores[idxMajor],
    )

    for index in order:
        angle, offset = maxima[index]
        angle = angle * fine_anglestep - angle_range + raw_angle
        offset += raw_offset - offset_range
        if direction == "v" and 45 < angle < 135:
            major_axis = Axis(
                angle=angle,
                radius=(size - offset) / 2,
                score=peak_scores[index],
            )
            break
        else:
            if 45 > angle or angle < 135:
                major_axis = Axis(
                    angle=angle,
                    radius=(size - offset) / 2,
                    score=peak_scores[index],
                )
                break
    end_timer = time.perf_counter()
    print(f"Compute PRST in {end_timer - start_timer:0.4f} seconds")
    print(f"score: {major_axis.score}, angle: {major_axis.angle}, normalized radius: {major_axis.radius / size}")

    # draw all axis
    if debug:
        allAxis = []
        for idx in order:
            angle, offset = maxima[idx]
            angle = angle * fine_anglestep - angle_range + raw_angle
            offset += raw_offset - offset_range
            axis = Axis(
                angle=angle,
                radius=(size - offset) / 2,
                score=peak_scores[idx],
            )
            allAxis.append(axis)
        for i in range(len(allAxis)):
            axis = allAxis[i]
            plt.subplot(2, math.ceil(len(allAxis) / 2.), i + 1)
            plt.title("%d " % axis.score)
            imgplot = plt.imshow(draw_axes(image_roi[..., 0] * 255, image_center, axis))
            imgplot.set_interpolation('nearest')
        plt.show()

        plt.subplot(221)
        imgplot = plt.imshow(image_roi)
        imgplot.set_interpolation('nearest')
        plt.subplot(222)
        plt.imshow(symmetryMap)
        plt.subplot(223)
        plt.title("%d " % major_axis.score)
        imgplot = plt.imshow(draw_axes(image_roi[..., 0] * 255, image_center, major_axis), cmap="gray")
        imgplot.set_interpolation('nearest')
        plt.show()

    return image_center, major_axis

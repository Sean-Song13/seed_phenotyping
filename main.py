import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import time
from skimage.draw import line_aa, line

from prst import PlanarReflectiveSymmetryTransform
from utils.util import draw_axes, get_axis, orient


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200, 200, 300, 300)
    win.setWindowTitle("Hi PyQt5")

    label = QtWidgets.QLabel(win)
    label.setText("first label")
    label.move(50, 50)

    win.show()
    sys.exit(app.exec_())


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def dis_p2line(x, start, end):
    line_vec = np.subtract(end, start)
    p_vec = x - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    p_vec_scaled = 1.0 / line_len * p_vec
    inner_product = np.dot(line_unit, p_vec_scaled)
    inner_product = np.clip(inner_product, 0.0, 1.0)
    nearest = start + line_vec * inner_product
    distance = np.linalg.norm(x - nearest)
    return distance, nearest


"""
x: sample point
M: model defined by points (x1, y1), (x2, y2)...
"""


def min_distance(x, M):
    min_dis = float('inf')
    for i in range(len(M) - 1):
        min_dis = min(min_dis, dis_p2line(x, M[i][0], M[i + 1][0])[0])
    return min(min_dis, dis_p2line(x, M[-1][0], M[0][0])[0])


"""
x: sample point
M: model defined by points (points array)
theta: width
"""


def gaussian(x, M, theta):
    g = np.exp(-pow(min_distance(x, M) / theta, 2))
    return 0 if g < 0.01 else g


def simple_gaussian(dis, theta):
    g = np.exp(-pow(dis / theta, 2))
    return 0 if g < 0.01 else g


def gaussian_edge(start, end, theta, gaussian_map):
    padding = int(np.ceil(np.linalg.norm(end - start) + theta * 2))
    x_min, x_max = min(start[0], end[0]), max(start[0], end[0])
    y_min, y_max = min(start[1], end[1]), max(start[1], end[1])
    window_width = x_max - x_min + 4 * padding
    window_height = y_max - y_min + 4 * padding
    window_origin = np.array([x_min, y_min]) - 2 * padding
    for dx in range(window_width + 1):
        for dy in range(window_height + 1):
            target = window_origin + [dx, dy]
            if target[1] >= gaussian_map.shape[0] or target[0] >= gaussian_map.shape[1]:
                continue
            if target[1] < 0 or target[0] < 0:
                continue
            g = simple_gaussian(dis_p2line(target, start, end)[0], theta)
            previous_g = gaussian_map[target[1]][target[0]]
            if previous_g != 0:
                gaussian_map[target[1]][target[0]] = max(g, previous_g)
            else:
                gaussian_map[target[1]][target[0]] = g


def gaussian_edge_opt(start, end, theta, gaussian_map):
    corners = get_padding_corners(start, end, theta)
    pixel = rasterization(corners)
    for p in pixel:
        g = simple_gaussian(dis_p2line(p, start, end)[0], theta)
        if p[1] >= gaussian_map.shape[0] or p[0] >= gaussian_map.shape[1]:
            continue
        previous_g = gaussian_map[p[1]][p[0]]
        if previous_g != 0:
            gaussian_map[p[1]][p[0]] = max(g, previous_g)
        else:
            gaussian_map[p[1]][p[0]] = g


def point_2_img(M, theta, y, x):
    gaussian_map = np.zeros((y, x))
    for i in range(gaussian_map.shape[0]):
        for j in range(gaussian_map.shape[1]):
            gaussian_map[i][j] = gaussian(np.array([j, i]), M, theta)
    img = np.zeros((y, x, 3))
    img[:, :, 0] = gaussian_map
    img[:, :, 1] = gaussian_map
    img[:, :, 2] = gaussian_map

    return img


def point_2_img_opt(M, theta, y, x):
    gaussian_map = np.zeros((y, x))
    num_M = len(M)
    for i in range(num_M):
        gaussian_edge(M[i][0], M[(i + 1) % num_M][0], theta, gaussian_map)
    img = np.zeros((y, x, 3))
    img[:, :, 0] = gaussian_map
    img[:, :, 1] = gaussian_map
    img[:, :, 2] = gaussian_map

    return img


def point_2_img_opt_opt(M, theta, y, x):
    gaussian_map = np.zeros((y, x))
    num_M = len(M)
    for i in range(num_M):
        gaussian_edge_opt(M[i][0], M[(i + 1) % num_M][0], theta, gaussian_map)
    img = np.zeros((y, x, 3))
    img[:, :, 0] = gaussian_map
    img[:, :, 1] = gaussian_map
    img[:, :, 2] = gaussian_map

    return img


def rasterization(corners):
    corners = np.round(corners).astype(int)
    cy = corners[:, 1]
    # cy = np.rint(cy).astype(int)
    target_pixel = []
    for y in range(min(cy), max(cy) + 1):
        row = find_intersections(y, corners)
        target_pixel.extend(row)
    return np.array(target_pixel)


def find_intersections(y, corners):
    n = corners.shape[0]
    target_edge = []
    for i in range(n):
        p, q = corners[i % n], corners[(i + 1) % n]
        if p[1] <= y <= q[1] or q[1] <= y <= p[1]:
            target_edge.append([p, q])

    if len(target_edge) < 1:
        return []

    bound = []
    for edge in target_edge:
        start = edge[0]
        end = edge[1]
        vec = np.subtract(end, start)
        if vec[1] == 0:
            bound.extend([start[0], end[0]])
        else:
            intersection = np.add(start, vec * (y - start[1]) / vec[1])
            intersection = np.round(intersection).astype(int)[0]
            bound.append(intersection)
    return np.array([[x, y] for x in range(min(bound), max(bound) + 1)])


# return four vertex points
def get_padding_corners(start, end, theta):
    length = np.linalg.norm(end - start)
    edge_vec = end - start
    sin_v = edge_vec[1] / length
    cos_v = edge_vec[0] / length

    c1 = np.add(start, theta * np.array([sin_v - cos_v, -cos_v - sin_v]))
    c2 = np.add(start, theta * np.array([-cos_v - sin_v, cos_v - sin_v]))
    c3 = np.add(end, theta * np.array([cos_v - sin_v, sin_v + cos_v]))
    c4 = np.add(end, theta * np.array([cos_v + sin_v, sin_v - cos_v]))
    corners = np.array([c1, c2, c3, c4])
    return corners


def longest_edge(M):
    num_M = len(M)
    max_length = 0
    for i in range(num_M):
        edge_vec = M[(i + 1) % num_M][0] - M[i][0]
        length = np.linalg.norm(edge_vec)
        if length > max_length:
            max_length = length
    return max_length


def image_preprocessing(image, resize=500, debug=False):
    scale_x = float(resize) / image.shape[1]
    scale_y = float(resize) / image.shape[0]
    scale = min(scale_x, scale_y)

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # visualize the binary image
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    if debug:
        plt.subplot(221)
        plt.imshow(img_gray, cmap="gray")
        plt.title("gray")
        plt.subplot(222)
        plt.imshow(thresh, cmap="gray")
        plt.title("binary")
        plt.subplot(223)
        plt.imshow(opening, cmap="gray")
        plt.title("opening")
        plt.subplot(224)
        plt.imshow(closing, cmap="gray")
        plt.title("closing")
        plt.show()

    return closing


def generate_contour_img(roi, theta=2.0, debug=False):
    start = time.perf_counter()
    img, contours, hierarchy = cv2.findContours(image=roi, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    maxlength = 0
    longest_contour = contours[0]
    for contour in contours:
        if len(contour) > maxlength:
            maxlength = len(contour)
            longest_contour = contour
    found_contour = time.perf_counter()
    print(f"found contour points {found_contour - start:0.4f} seconds\nlength of contour: {len(longest_contour)}")

    img = point_2_img_opt_opt(longest_contour, theta, roi.shape[0], roi.shape[1])
    end = time.perf_counter()
    print(f"Generate outline image in {end - start:0.4f} seconds")
    # print(f"Resolution: {img.shape[1]}, {img.shape[0]}")
    if debug:
        plt.imshow(img)
        plt.title("contour")
        plt.show()
    return img, longest_contour


def width_analysis(contour, size, center, axis, poly=False):
    start, end = get_axis(size, center, axis)
    upper_contour = list(filter(lambda p: orient(start, end, p[0]) < 0, contour))
    lower_contour = list(filter(lambda p: orient(start, end, p[0]) > 0, contour))

    upper_distance, upper_position = get_width_position(upper_contour, start, end)
    lower_distance, lower_position = get_width_position(lower_contour, start, end)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(upper_position, upper_distance, color='orange', label='upper')
    axes.plot(lower_position, lower_distance, color='blue', label='lower')
    plt.legend()
    plt.show()

    if poly:
        upper_poly = np.polyfit(upper_position, upper_distance, 7)
        upper_distance = np.poly1d(upper_poly)(upper_position)
        lower_poly = np.polyfit(lower_position, lower_distance, 7)
        lower_distance = np.poly1d(lower_poly)(lower_position)
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(upper_position, upper_distance, color='orange', label='upper')
        axes.plot(lower_position, lower_distance, color='blue', label='lower')
        plt.legend()
        plt.show()


def get_width_position(contour, start, end):
    dis_and_point = [dis_p2line(point[0], start, end) for point in contour]
    width = np.array([d[0] for d in dis_and_point])
    intersections = np.array([d[1] for d in dis_and_point])

    sort_idx = intersections[:, 0].argsort()
    intersections = intersections[sort_idx]
    width = width[sort_idx]
    head = intersections[0]
    tail = intersections[-1]

    length = np.linalg.norm(np.subtract(head, tail))
    position = [np.linalg.norm(np.subtract(p, head)) / length for p in intersections]

    return width, position

def find_difference(upper, lower):
    pass

def find_peek():
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    debug = True
    root = r"E:\Study\CSE598\seed_img"
    target = ["2N", "6N"]
    sizes = [160, 320, 480]
    size = 160 * 3
    theta = size / 80
    result_file = f"{size}N_V_{int(theta)}.txt"

    # for t in target:
    #     group_path = os.path.join(root, t, "CHARRED")
    #     if not os.path.isdir(group_path):
    #         continue
    #     groups = os.listdir(group_path)
    #     for g in groups:
    #         image_path = os.path.join(group_path, g, "Ventral")
    #         if not os.path.isdir(image_path):
    #             continue
    #         images = os.listdir(image_path)
    #         for name in images:
    #             full_path = os.path.join(image_path, name)
    #             if not os.path.isfile(full_path):
    #                 continue
    #             image = cv2.imread(full_path)
    #             if image.shape[0] < 100:
    #                 continue
    #             if not debug:
    #                 print(f"processing: {full_path}")
    #                 roi = image_preprocessing(image, resize=size)
    #                 contour_img = generate_contour_img(roi, theta=theta)
    #                 center, major, minor = PlanarReflectiveSymmetryTransform(contour_img)
    #                 with open(result_file, "a") as f:
    #                     f.write("{0:.6f}\n".format(major.score))
    #                 print("\n")
    #
    #     with open(result_file, "a") as f:
    #         f.write("\n")
    #
    # print(f"result in {result_file}\n")

    image = cv2.imread(r"6N7_1.V.jpg")
    roi = image_preprocessing(image, resize=size, debug=debug)
    contour_img, contour = generate_contour_img(roi, theta=theta, debug=debug)
    center, axis = PlanarReflectiveSymmetryTransform(contour_img, debug=debug)

    width_analysis(contour, size, center, axis, poly=True)
    # out = np.zeros(contour_img.shape[:2])
    # start, end = get_axis(size, center, axis)
    # coords = line(start[1], start[0], end[1], end[0])
    # for r, c in zip(*coords):
    #     if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
    #         out[r, c] = 255
    # for point in contour:
    #     point = point[0]
    #     out[point[1], point[0]] = 255
    # plt.imshow(out, cmap="gray")
    # plt.show()

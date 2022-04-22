import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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
        p, q = corners[i % n], corners[(i+1) % n ]
        if p[1] <= y <= q[1] or q[1] <= y <= p[1]:
            target_edge.append([p, q])

    if len(target_edge) < 1:
        return []

    bound = []
    for edge in target_edge:
        start = edge[0]
        end = edge[1]
        vec = np.subtract(end , start)
        intersection = np.add(start, vec * (y - start[1]) / vec[1])
        intersection = np.round(intersection).astype(int)[0]
        bound.append(intersection)
    return np.array([[x, y] for x in range(min(bound), max(bound) + 1)])


# return four vertex points
def padding_corners(start, end, theta):
    theta *= 1
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

def main():
    x1, y1 = [5], [4]
    x2, y2 = [9, 3], [1, 5]
    plt.plot(x2, y2, marker='o')
    theta = 1
    corners = padding_corners(np.array([9, 1]), np.array([3, 5]), theta)
    cx, cy = corners[:, 0], corners[:, 1]
    cx_r, cy_r = np.round(cx), np.round(cy)

    pixel = rasterization(corners)
    px, py = pixel[:, 0], pixel[:, 1]
    plt.scatter(cx_r, cy_r, marker='.', color='orange')
    plt.scatter(cx, cy, marker='.', color='red')
    plt.scatter(px, py, marker='x', color='green')
    plt.xticks(range(-2, 15))
    plt.yticks(range(-2, 15))
    plt.show()


if __name__ == '__main__':
    data = {}
    cate = ""
    with open("480N_V_0414.txt") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                break
            if line.startswith('score'):
                numbers_pat = r"(\d+\.\d+\b)"
                numbers = re.findall(numbers_pat, line)
                score = float(numbers[0])
                peek = float(numbers[1])
                peek_diff = float(numbers[2])
                curve_diff = float(numbers[3])
                hausdorff = float(numbers[4])
                if score > 0.75:
                    data[cate].append(score)
            else:
                cate = line[:-1]
                data[cate] = []

    print("# 2N mean: {0:.6f} std: {1:.6f} number:{2}".format(np.mean(data["2N"]), np.std(data["2N"]), len(data["2N"])))
    print("# 6N mean: {0:.6f} std: {1:.6f} number:{2}".format(np.mean(data["6N"]), np.std(data["6N"]), len(data["6N"])))

    sns.set_theme()
    sns.distplot(data["2N"], label="2N",bins=10, kde=True)
    sns.distplot(data["6N"], label="6N",bins=10, kde=True)
    plt.title('score')
    plt.legend()
    plt.show()

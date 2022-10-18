import math

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.datasets import make_blobs


def sqr_distance(dot1: np.ndarray, dot2: np.ndarray):
    axis_num = dot1.shape[0]
    result = 0
    for i in range(axis_num):
        result += (dot1[i] - dot2[i]) ** 2
    return result


def rebalance(centers: np.ndarray, centers_belong: np.ndarray, data: np.ndarray):
    new_centers = np.zeros(centers.shape)
    dots_in_center = np.zeros(centers.shape[0])
    new_center_belong = np.zeros(data.shape[0])
    for i, dot in enumerate(data):
        min_dist = 0
        min_center = -1
        for j, c in enumerate(centers):
            dist = sqr_distance(c, dot)
            if dist < min_dist or min_center == -1:
                min_dist = dist
                min_center = j
        new_center_belong[i] = min_center
        new_centers[min_center] += dot
        dots_in_center[min_center] += 1
    for i, c in enumerate(new_centers):
        new_centers[i] = c / dots_in_center[i]
    return new_centers, new_center_belong


def draw_results(centers_history, centers_belong_history, data):
    fig, ax = plt.subplots(math.ceil(len(centers_history) / 2), 2)
    for i, c in enumerate(centers_history):
        colors = [['r', 'g', 'y', 'c', 'm', 'k'][math.ceil(color)] for color in centers_belong_history[i]]
        ax[i // 2, i % 2].scatter(data[:, 0], data[:, 1], color=colors)
        ax[i // 2, i % 2].scatter(centers_history[i][:, 0], centers_history[i][:, 1], color='b', marker='X')
    plt.show()


def mean_distance(centers, centers_belong, data):
    result = 0
    for i, dot in enumerate(data):
        center = centers[math.ceil(centers_belong[i])]
        result += sqr_distance(center, dot)
    return result


def kmeans(data: np.ndarray, centers_num: int):
    centers = list()
    centers_n = set()
    i = 0
    while i < centers_num:
        dot_n = random.randint(0, data.shape[0] - 1)
        dot = data[dot_n]
        if dot_n not in centers_n:
            centers_n.add(dot_n)
            centers.append(dot)
            i += 1
    centers = np.array(centers)
    centers_belong = np.zeros(data.shape[0])
    centers_stopped = False
    centers_history = [centers]
    centers_belong_history = [centers_belong]
    while not centers_stopped:
        new_centers, new_center_belong = rebalance(centers, centers_belong, data)
        centers_history.append(new_centers)
        centers_belong_history.append(new_center_belong)
        if not (new_centers - centers).any():
            centers_stopped = True
        centers = new_centers
        centers_belong = new_center_belong
    return mean_distance(centers, centers_belong, data), centers_history, centers_belong_history


if __name__ == '__main__':
    dataset = make_blobs(200)[0]
    k = 1
    last_mean_dist, center_hist, center_belong_hist = kmeans(dataset, k)
    k += 1
    while k < 6:
        mean_dist, new_center_hist, new_center_belong_hist = kmeans(dataset, k)
        print(mean_dist, last_mean_dist, mean_dist / last_mean_dist)
        if mean_dist / last_mean_dist > 0.6:
            break
        center_hist = new_center_hist
        center_belong_hist = new_center_belong_hist
        last_mean_dist = mean_dist
        k += 1
    draw_results(center_hist, center_belong_hist, dataset)
    print(k - 1)

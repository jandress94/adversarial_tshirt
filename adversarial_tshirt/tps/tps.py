#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# tps.py
# Author: Jim Andress
# Created: 2020-08-30

import numpy as np
import scipy
from sklearn.metrics import pairwise_distances

from PIL import Image


def u(r):
    return r * r * np.log(r, out=np.zeros_like(r, dtype=np.float32), where=(r != 0))


def get_spline_fn(a, p, w):
    def fn(x, y):
        p_dists = np.linalg.norm(p - np.array([x, y]), axis=1)
        return np.dot(np.array([1, x, y]), a) + np.dot(w, u(p_dists))

    return fn


def solve_tps(p, h):
    p_aug = np.ones((p.shape[0], 3), dtype=np.float32)
    p_aug[:, 1:] = p

    y = np.concatenate([h, np.zeros((3, ), dtype=np.float32)])

    r = pairwise_distances(p)
    k = u(r)

    l = np.block([[k, p_aug], [p_aug.T, np.zeros((3, 3), dtype=np.float32)]])

    x = scipy.linalg.solve(l, y, assume_a='sym')

    return x[:-3], x[-3:]


def solve_img_tps(orig_img, control_pts_start, control_pts_end, num_pin_pts=5):
    if num_pin_pts is not None and num_pin_pts >= 2:
        h, w, _ = orig_img.shape

        pin_delta_h = (h - 1) // (num_pin_pts - 1)
        pin_delta_w = (w - 1) // (num_pin_pts - 1)

        corners = []
        for i in range(num_pin_pts - 1):
            corners += [
                [0, i * pin_delta_w],
                [(num_pin_pts - 1) * pin_delta_h, (i + 1) * pin_delta_w],
                [(i + 1) * pin_delta_w, 0],
                [i * pin_delta_w, (num_pin_pts - 1) * pin_delta_w]
            ]

        corners = np.array(corners)

        control_pts_start = np.concatenate([control_pts_start, corners])
        control_pts_end = np.concatenate([control_pts_end, corners])

    delta_x = control_pts_start[:, 0] - control_pts_end[:, 0]
    delta_y = control_pts_start[:, 1] - control_pts_end[:, 1]

    w_x, a_x = solve_tps(control_pts_end, delta_x)
    w_y, a_y = solve_tps(control_pts_end, delta_y)

    delta_x_fn = get_spline_fn(a_x, control_pts_end, w_x)
    delta_y_fn = get_spline_fn(a_y, control_pts_end, w_y)

    new_img = np.zeros_like(orig_img)

    for x in range(orig_img.shape[1]):
        for y in range(orig_img.shape[0]):
            new_x = int(round(x + delta_x_fn(x, y)))
            new_y = int(round(y + delta_y_fn(x, y)))

            if new_x >= 0 and new_x < orig_img.shape[1] and new_y > 0 and new_y < orig_img.shape[0]:
                new_img[y, x, :] = orig_img[new_y, new_x, :]

    return Image.fromarray(new_img)




# ORIG_IMG_PATH = 'data/img/head.jpg'
# NEW_IMG_PATH = 'data/img/head_warped.jpg'

ORIG_IMG_PATH = 'data/img/head.jpg'
NEW_IMG_PATH = 'data/img/head_warped.jpg'


img = Image.open(ORIG_IMG_PATH)
img.load()
img_data = np.asarray(img)

# solve_img_tps(img_data, np.array([[100, 200], [300, 400], [500, 600]]), np.array([[90, 210], [320, 380], [470, 630]]))


# new_img = solve_img_tps(img_data, np.array([[900, 820], [980, 780], [1040, 730]]), np.array([[850, 800], [980, 780], [1040, 680]]))
new_img = solve_img_tps(img_data, np.array([[970, 700], [870, 160], [420, 570]]), np.array([[860, 600], [1050, 120], [350, 760]]))
# new_img = solve_img_tps(img_data, np.array([[97, 70]]), np.array([[86, 60]]))
new_img.save(NEW_IMG_PATH)




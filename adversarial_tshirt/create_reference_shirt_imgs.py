#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# create_reference_shirt_imgs.py
# Author: Jim Andress
# Created: 2020-08-31

import numpy as np
from PIL import Image


REF_SHIRT_CHECKERBOARD_IMG_FILE_PATH = 'data/reference_shirt/checkerboard.png'
REF_SHIRT_COLOR_MAP_FILE_PATH = 'data/reference_shirt/color_map.png'


def create_checkerboard_img(num_squares_x=9, num_squares_y=17, square_sz_px=100):
    Image.fromarray(np.add(*np.meshgrid(range(num_squares_x), range(num_squares_y))) % 2 > 0) \
        .resize((num_squares_x * square_sz_px, num_squares_y * square_sz_px)) \
        .save(REF_SHIRT_CHECKERBOARD_IMG_FILE_PATH)


def create_color_map_img(num_steps_per_dim=9, split_dim=2, num_square_rows=3, square_sz_px=100):
    steps = np.minimum(np.linspace(0, 256, num_steps_per_dim), 255).astype(np.uint8)

    x, y, z = np.meshgrid(steps, steps, steps)

    # create rgb values as n x n x n x 3 tensor
    color_cube = np.stack([x, y, z], axis=-1)

    # slice along requested dimension and join into long row of slices
    color_cube = np.split(color_cube, color_cube.shape[split_dim], axis=split_dim)
    color_cube = np.concatenate(color_cube, axis=1)
    color_cube = np.squeeze(color_cube)

    # split long row into requested number of rows
    color_cube = np.split(color_cube, num_square_rows, axis=1)
    color_cube = np.concatenate(color_cube, axis=0)

    # convert to PIL image and size up
    img = Image.fromarray(color_cube)
    img = img.resize((img.width * square_sz_px, img.height * square_sz_px), resample=Image.NEAREST)
    img.save(REF_SHIRT_COLOR_MAP_FILE_PATH)


if __name__ == '__main__':
    create_checkerboard_img()
    create_color_map_img(num_steps_per_dim=12, num_square_rows=4, square_sz_px=25)

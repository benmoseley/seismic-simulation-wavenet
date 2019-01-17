#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""

import numpy as np


def fig2rgb_array(fig, expand=False):
    fig.canvas.draw()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(shape)
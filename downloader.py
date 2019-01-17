#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""


import tqdm
import requests

def get_url(url, file_path):
    "download a file from url, using requests and reporting progress with tqdm progress bar"
    
    response = requests.get(url, stream=True)
    n_bytes = int(response.headers["Content-Length"])
    bar_format = '{l_bar}{bar}| {elapsed}, {rate_fmt}'
    with open(file_path, "wb") as f:
        with tqdm.tqdm(total=n_bytes,
                  unit="b",
                  unit_scale=True,
                  unit_divisor=1e6,
                  desc="Downloading %s %.0fMb"%(url.split("/")[-1], n_bytes/1e6),
                  ncols=80,
                  bar_format=bar_format) as pbar:
            for chunk in response.iter_content(chunk_size=1024):# chunk_size in bytes
                f.write(chunk)
                pbar.update(1024)

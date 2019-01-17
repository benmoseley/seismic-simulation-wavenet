#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:57:30 2019

@author: bmoseley
"""

import os
import shutil
import glob

## GENERAL DIRECTORY HELPER FUNCTIONS

def get_dir(directory):
    """
    Creates the given directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.
    """
    # important! if None passed to os.listdir, current directory is wiped (!)
    if not os.path.isdir(directory): raise Exception("%s is not a directory"%(directory))
    if type(directory) != str: raise Exception("string type required for directory: %s"%(directory))
    if directory in ["..",".", "","/","./","../","*"]: raise Exception("trying to delete current directory, probably bad idea?!")
    
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def clear_files(glob_expression):
    """
    Removes all files matching glob expression.
    """
    for f in glob.glob(glob_expression): 
        if os.path.isfile(f):
            os.remove(f)

def remove_dir(directory):
    """
    Recursively removes directory.
    """
    # important!
    if not os.path.isdir(directory): raise Exception("%s is not a directory"%(directory))
    if type(directory) != str: raise Exception("string type required for directory: %s"%(directory))
    if directory in ["..",".", "","/","./","../", "*"]: raise Exception("trying to delete current directory, probably bad idea?!")
    
    shutil.rmtree(directory)
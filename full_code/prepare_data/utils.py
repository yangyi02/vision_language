# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:08:21 2015

@author: yi
"""

import os
from os import path
import io

def list_folders(folders, list_file):
    """
    Given any number of folders, merge all file names in the folders to a list.
    """
    #if type(folders) is not list:
    if not isinstance(folders, list):
        folders = [folders]
    files = []
    for folder in list(set(folders)):
        for item in os.listdir(folder):
            if path.isfile(path.join(folder, item)):
                files.append(path.join(folder, item))
    handle = io.open(list_file, 'wb')
    handle.writelines(['%s\n' % item for item in files])

def list_files(files, list_file):
    """
    Given a list of files, write file names to a list
    """
    handle = io.open(list_file, 'wb')
    handle.writelines(['%s\n' % item for item in list(files)])

def merge_lists(in_list_files, out_list_file):
    """
    Given a set of list files, merge them into one list file
    """
    files = []
    for list_file in list(in_list_files):
        lines = list(io.open(list_file, 'rb').readlines())
        for line in lines:
            line = line.strip()
            if line in files:
                continue
            files.append(line)
    handle = io.open(out_list_file, 'wb')
    handle.writelines(['%s\n' % item for item in files])

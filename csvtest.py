# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:57:03 2022

@author: muskr
"""
import os
import shutil

start_path = './srip2022-simulator/final/'
end_path = './srip2022-simulator/all_files/'


for file_name in os.listdir(start_path):
    seq = file_name.split('_')[0]
    if not os.path.isdir(end_path + str(seq)) and seq != 'data.csv':
        os.mkdir(end_path + str(seq))
    if not os.path.isfile(end_path + str(seq) + str(file_name)):
        shutil.copy(start_path + str(file_name), end_path + str(seq))
########################################################################################################################
# source code for downloading dataset found at https://compneuro.net/ (Cramer et al., 2020)
# Authors: Benjamin Cramer & Friedemann Zenke. Licensed under a Creative Commons Attribution 4.0 International License
# https://creativecommons.org/licenses/by/4.0/ slightly modified
########################################################################################################################

import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cache_dir=os.path.expanduser("./data")
cache_subdir="hdspikes"
print("Using cache dir: %s"%cache_dir)

# The remote directory with the data files
base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote
response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
data = response.read()
lines = data.decode('utf-8').split("\n")
file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }

def get_and_gunzip(origin, filename, md5hash=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path=gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

# Download the Spiking Heidelberg Digits (SHD) dataset
files = ["shd_test.h5.gz", "shd_train.h5.gz"]

hdf5_file_path = []
for fn in files:
    origin = "%s/%s"%(base_url,fn)
    _hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
    print(_hdf5_file_path)
    hdf5_file_path.append(_hdf5_file_path)

########################################################################################################################
# source for preproccessing:
# https://github.com/byin-cwi/Efficient-spiking-networks/blob/main/SHD/generate_dataset.py (Yin et al., 2021)
#
# MIT License
#
# Copyright (c) 2021 byin-cwi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
########################################################################################################################
import tables
import numpy as np

import torch
# files = ['data/hdspikes/shd_test.h5', 'data/hdspikes/shd_train.h5']

fileh = tables.open_file(hdf5_file_path[0], mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.labels

# This is how we access spikes and labels
index = 0
print("Times (ms):", times[index], max(times[index]))
print("Unit IDs:", units[index])
print("Label:", labels[index])


def binary_image_readout(times, units, dt=1e-3):
    img = []
    N = int(1 / dt)
    for i in range(N):

        # get idx that occur appears in times
        idxs = np.argwhere(times <= i * dt).flatten()
        # get the channels that were active at the time step i
        vals = units[idxs]
        vals = vals[vals > 0]

        # spike for those channels
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)

    return np.array(img)



def generate_dataset(file_name,datafile, dt=1e-3):

    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ", len(times))
    X = []
    y = []
    for i in range(len(times)):
        tmp = binary_image_readout(times[i], units[i], dt=dt)
        torch.save([torch.Tensor(tmp).transpose(0,1), torch.Tensor([labels[i],])],datafile.format(i))
        print('\r test i:',i,end='')
    return 0.0

import os

# Get absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct dataset directories relative to the script
train_dir = os.path.join(script_dir, '../../datasets/shd/train')
test_dir = os.path.join(script_dir, '../../datasets/shd/test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_filename = os.path.join(train_dir, '{}.pt')
test_filename = os.path.join(test_dir, '{}.pt')


a = generate_dataset(hdf5_file_path[0],test_filename, dt=4e-3)
a = generate_dataset(hdf5_file_path[1],train_filename, dt=4e-3)
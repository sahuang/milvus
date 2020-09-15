import csv
import os
import logging
import pdb
import time
import random
from multiprocessing import Process
import numpy as np
import sklearn.preprocessing
from milvus import DataType, Milvus

SIFT_VECTORS_PER_FILE = 100000
MAX_NQ = 10001
FILE_PREFIX = "binary_"
SIFT_SRC_DATA_DIR = '/home/sift1b/'

def get_vectors_from_binary(nq):
    file_name = SIFT_SRC_DATA_DIR + "query.npy"
    data = np.load(file_name)
    return data[0:nq].tolist()

a = get_vectors_from_binary(1000)

with open("query.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(a)

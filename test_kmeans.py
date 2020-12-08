import os
import h5py
import numpy as np
from milvus import Milvus, DataType
import sys
from pprint import pprint
import csv
import time

def get_dataset(hdf5_file_path):
    if not os.path.exists(hdf5_file_path):
        raise Exception("%s not existed" % hdf5_file_path)
    dataset = h5py.File(hdf5_file_path)
    return dataset

def get_recall_value(true_ids, result_ids):
    """
    Use the intersection length
    """
    sum_radio = 0.0
    for index, item in enumerate(result_ids):
        tmp = set(true_ids[index]).intersection(set(item))
        sum_radio = sum_radio + len(tmp) / len(item)
    return round(sum_radio / len(result_ids), 3)

def get_ids(result):
    idss = result._entities.ids
    ids = []
    len_idss = len(idss)
    len_r = len(result)
    top_k = len_idss // len_r
    for offset in range(0, len_idss, top_k):
        ids.append(idss[offset: min(offset + top_k, len_idss)])
    return ids

_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)
nb = 1000000
nq = 100
SIFT_PATH = '/home/ann_hdf5/sift-128-euclidean.hdf5'
GIST_PATH = '/home/ann_hdf5/gist-960-euclidean.hdf5'

'''
This script will take several arguments.

e.g. 
1) python3 test_kmeans.py SIFT 10 IVF_FLAT
2) python3 test_kmeans.py GIST 2 IVF_PQ 16

argv[0]: This file name, test_kmeans.py
argv[1]: Dataset type. SIFT(d=128) and GIST(d=960)
argv[2]: Number of segments. {1, 2, 10}
argv[3]: Index Type. {IVF_FLAT, IVF_SQ8, IVF_PQ}
argv[4]: If exists, this is M in IVF_PQ
'''
if len(sys.argv) < 4:
    raise Exception("Too few arguments!")
elif len(sys.argv) > 5:
    raise Exception("Too many arguments!")

combinations = []
for nlist in [1024, 2048, 4096]:
    for nprobe in [1, 2, 4, 8, 16, 32]:
        for topk in [10, 50, 100]:
            combinations.append((nlist, nprobe, topk))

try:
    # Read input
    if sys.argv[1] == 'SIFT':
        dataset = get_dataset(SIFT_PATH)
        dim = 128
    else:
        dataset = get_dataset(GIST_PATH)
        dim = 960
    collection_name = sys.argv[1] + '_' + sys.argv[2]
    index_type = sys.argv[3]
    recalls = []
    csv_name = 'Early_' + index_type + '_' + collection_name + '.csv'
    with open(csv_name,'a') as fd:
        fd.write("{},{},{},{},{},{},{},{},{},{}\n".format(
            'nlist','nprobe','topK',
            'niter','objective','imbalance','training time (s)',
            'quantization time (ms)', 'data search time (ms)', 'recall'
        ))
    for c in combinations:
        nlist = c[0]
        nprobe = c[1]
        topK = c[2]
        if len(sys.argv) == 5:
            M = int(sys.argv[4])
        print("======Dataset: {}, Index Type: {}, nlist: {}, nprobe: {}, topK: {}======".format(collection_name, index_type, nlist, nprobe, topK))

        # Create collection, insert data, create index
        client.drop_index(collection_name, "embedding")
        client.create_index(collection_name, "embedding", {"index_type": index_type, "metric_type": "L2", "params": {"nlist": nlist}})
        pprint(client.get_collection_info(collection_name))
        print("==========")

        # Search
        query_embedding = np.array(dataset["test"][:nq])
        query_hybrid = {
            "bool": {
                "must": [{
                    "vector": {
                        "embedding": {"topk": topK,
                                    "query": query_embedding.tolist(),
                                    "metric_type": "L2",
                                    "params": {"nprobe": nprobe}}
                    }
                }]
            }
        }
        results = client.search(collection_name, query_hybrid)
        result_ids = get_ids(results)
        true_ids = np.array(dataset["neighbors"])
        acc_value = get_recall_value(true_ids[:nq, :topK].tolist(), result_ids)
        recalls.append(acc_value)
        print("Recall: {}".format(acc_value))

        # CSV operations
        '''
        fp = open('/tmp/server_file.txt', 'r')
        lines = fp.readlines()
        segments = int(sys.argv[2])
        niter = []
        train_times = []
        objectives = []
        imbalance = []
        quant_time = 0
        search_time = 0
        for loop in range(segments):
            niter.append(int(lines[4 * loop]))
            train_times.append(float(lines[4 * loop + 1]))
            objectives.append(float(lines[4 * loop + 2]))
            imbalance.append(float(lines[4 * loop + 3]))
            quant_time += float(lines[4 * segments + loop * 2])
            search_time += float(lines[4 * segments + loop * 2 + 1])
        with open(csv_name,'a') as fd:
            fd.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                nlist,nprobe,topK,
                "_".join([str(x) for x in niter]),"_".join([str(x) for x in objectives]),
                "_".join([str(x) for x in imbalance]),"_".join([str(x) for x in train_times]),quant_time,search_time,
                acc_value
            ))
        os.system("rm -rf /tmp/server_file.txt")
        '''
        time.sleep(1)
except Exception as e:
    raise Exception(e)


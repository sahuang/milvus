import os
import h5py
import numpy as np
from milvus import Milvus, DataType
import sys
from pprint import pprint

def get_dataset(hdf5_file_path):
    if not os.path.exists(hdf5_file_path):
        raise Exception("%s not existed" % hdf5_file_path)
    dataset = h5py.File(hdf5_file_path)
    return dataset

_HOST = '127.0.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)
nb = 1000000
SIFT_PATH = '/home/ann_hdf5/sift-128-euclidean.hdf5'
GIST_PATH = '/home/ann_hdf5/gist-960-euclidean.hdf5'

try:
    # Read input
    dataset_sift = get_dataset(SIFT_PATH)
    dim_sift = 128
    dataset_gist = get_dataset(GIST_PATH)
    dim_gist = 960
    for segments in [1, 2, 10]:
        collection_name_sift = 'SIFT_' + str(segments)
        collection_name_gist = 'GIST_' + str(segments)
        # Create collection, insert data
        row_in_segment = nb // segments
        collection_param_sift = {
            "fields": [
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dim_sift}},
            ],
            "segment_row_limit": row_in_segment,
            "auto_id": False
        }
        collection_param_gist = {
            "fields": [
                {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dim_gist}},
            ],
            "segment_row_limit": row_in_segment,
            "auto_id": False
        }
        client.create_collection(collection_name_sift, collection_param_sift)
        client.create_collection(collection_name_gist, collection_param_gist)
        insert_vectors_sift = np.array(dataset_sift["train"]).tolist()
        insert_vectors_gist = np.array(dataset_gist["train"]).tolist()
        for loop in range(segments):
            start = loop * row_in_segment
            end = min((loop + 1) * row_in_segment, nb)
            if start < end:
                tmp_vectors_sift = insert_vectors_sift[start:end]
                tmp_vectors_gist = insert_vectors_gist[start:end]
                ids = [i for i in range(start, end)]
                hybrid_entities_sift = [
                    {"name": "embedding", "values": tmp_vectors_sift, "type": DataType.FLOAT_VECTOR},
                ]
                hybrid_entities_gist = [
                    {"name": "embedding", "values": tmp_vectors_gist, "type": DataType.FLOAT_VECTOR},
                ]
                res_ids_1 = client.insert(collection_name_sift, hybrid_entities_sift, ids)
                res_ids_2 = client.insert(collection_name_gist, hybrid_entities_gist, ids)
                assert res_ids_1 == ids
                assert res_ids_2 == ids
        client.flush([collection_name_sift, collection_name_gist])
        print("Total row count sift: {}".format(client.count_entities(collection_name_sift)))
        print("Total row count gist: {}".format(client.count_entities(collection_name_gist)))
    print(client.list_collections())
except Exception as e:
    raise Exception(e)
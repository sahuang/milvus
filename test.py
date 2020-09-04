import time
import sys
import string

sys.path.append(".")
import random
import time
from milvus import Milvus, DataType

_HOST = '127.0.0.1'
_PORT = '19530'

def main():

    dim = 64
    rows = 1000000
    letters = string.ascii_lowercase
    name = ''.join(random.choice(letters) for i in range(5))
    collection_name = 'example_collection_' + name

    milvus = Milvus(_HOST, _PORT)

    collection_param = {
        "fields": [
            {"field": "int_field", "type": DataType.INT32},
            {"field": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": dim}}
        ],
        "segment_row_count": 100000,
        "auto_id": False
    }
    milvus.create_collection(collection_name, collection_param)

    # for int field, we use uniform distribution [0, 1000), each value has 1000 rows.
    int_list = []
    for i in range(1000):
        for j in range(1000):
            int_list.append(i)
    vec = [[random.random() for _ in range(dim)] for _ in range(rows)]
    hybrid_entities = [
        {"field": "int_field", "values": int_list, "type": DataType.INT32},
        {"field": "vec", "values": vec, "type": DataType.FLOAT_VECTOR}
    ]

    ids = milvus.insert(collection_name, hybrid_entities, [i for i in range(rows)])
    milvus.flush([collection_name])
    print("Flush ... ")
    time.sleep(3)

    print("Create index ......")
    milvus.create_index(collection_name, "vec", {"index_type": "IVF_FLAT", "params": {"nlist": 1000}, "metric_type": "L2"})
    print("Create index done.")
    print()

    # experiment 1 - only few entities left. strategy 1 should be preferred.
    print("==========Experiment 1==========")
    print("1,000 out of 1,000,000 remain. Strategy 1 is preferred.")
    for strategy in range(1, 4):
        print("==========Strategy " + str(strategy) + "==========")
        query_hybrid = {
            "bool": {
                "must": [
                    {
                        "range": {
                            "int_field": {"GT": -1, "LT": 1}
                        }
                    },
                    {
                        "vector": {
                            "vec": {
                                "topk": 10, "query": vec[1995: 2000], "params": {"nprobe": 50}
                            }
                        }
                    }
                ],
            },
            "strategy": strategy
        }

        print("Start searach ..")
        t0 = time.time()
        results = milvus.search(collection_name, query_hybrid)
        print("Time spent for search: " + str(time.time() - t0))
        for r in list(results):
            print("ids: ", r.ids)
            print("distances: ", r.distances)

    milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
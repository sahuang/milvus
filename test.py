import time
import sys
import string

sys.path.append(".")
import random
import time
from milvus import Milvus, DataType

_HOST = '127.0.0.1'
_PORT = '19530'

def get_recall_value(true_ids, result_ids):
    """
    Use the intersection length
    """
    sum_radio = 0.0
    for index, item in enumerate(result_ids):
        # tmp = set(item).intersection(set(flat_id_list[index]))
        tmp = set(true_ids[index]).intersection(set(item))
        sum_radio = sum_radio + len(tmp) / len(item)
        # logger.debug(sum_radio)
    return round(sum_radio / len(result_ids), 3)

def get_ids(result):
    ids = []
    for item in result:
        ids.append([entity.id for entity in item])
    return ids


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
            int_list.append(j)
    vec = [[random.random() for _ in range(dim)] for _ in range(rows)]
    hybrid_entities = [
        {"field": "int_field", "values": int_list, "type": DataType.INT32},
        {"field": "vec", "values": vec, "type": DataType.FLOAT_VECTOR}
    ]

    ids = milvus.insert(collection_name, hybrid_entities, [i for i in range(rows)])
    milvus.flush([collection_name])
    print("Flush ... ")
    time.sleep(1)

    print("Create index ......")
    milvus.create_index(collection_name, "vec", {"index_type": "IVF_FLAT", "params": {"nlist": 1024}, "metric_type": "L2"})
    print("Create index done.")
    print()

    res_ids = [0, 1, 2, 3]
    passed = False

    # experiment
    print("==========Experiment ==========")
    for strategy in [2, 2, 3, 1]:
        if passed:
            print("==========Strategy " + str(strategy) + "==========")
        query_hybrid = {
            "bool": {
                "must": [
                    {
                        "range": {
                            "int_field": {"GT": -1000, "LT": 2000}
                        }
                    },
                    {
                        "vector": {
                            "vec": {
                                "topk": 10, "query": vec[0: 1000], "params": {"nprobe": 500}
                            }
                        }
                    }
                ],
            },
            "strategy": strategy
        }

        if passed:
            print("Start search ..")
        t0 = time.time()
        results = milvus.search(collection_name, query_hybrid)
        if passed:
            print("Time spent for search: " + str(time.time() - t0))
        res_ids[strategy] = get_ids(results)
        passed = True


    # check recall rate
    print("Strategy 2 recall rate: " + str(get_recall_value(res_ids[1], res_ids[2])))
    print("Strategy 3 recall rate: " + str(get_recall_value(res_ids[1], res_ids[3])))

    milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
import os
import sys
import random
import pdb
import string
import struct
import logging
import time, datetime
import copy
import numpy as np
from sklearn import preprocessing
from milvus import Milvus, DataType

port = 19530
epsilon = 0.000001
default_flush_interval = 1
big_flush_interval = 1000
dimension = 128
segment_row_count = 5000
nb = 1000
default_float_vec_field_name = "float_vector"
default_binary_vec_field_name = "binary_vector"

# TODO:
all_index_types = [
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_SQ8_HYBRID",
    "IVF_PQ",
    "HNSW",
    # "NSG",
    "ANNOY",
    "BIN_FLAT",
    "BIN_IVF_FLAT"
]

default_index_params = [
    {"nlist": 1024},
    {"nlist": 1024},
    {"nlist": 1024},
    {"nlist": 1024},
    {"nlist": 1024, "m": 16},
    {"M": 48, "efConstruction": 500},
    # {"search_length": 50, "out_degree": 40, "candidate_pool_size": 100, "knng": 50},
    {"n_trees": 4},
    {"nlist": 1024},
    {"nlist": 1024}
]


def index_cpu_not_support():
    return ["IVF_SQ8_HYBRID"]


def binary_support():
    return ["BIN_FLAT", "BIN_IVF_FLAT"]


def delete_support():
    return ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_SQ8_HYBRID", "IVF_PQ"]


def ivf():
    return ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_SQ8_HYBRID", "IVF_PQ"]


def l2(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def ip(x, y):
    return np.inner(np.array(x), np.array(y))


def jaccard(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return 1 - np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())


def hamming(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return np.bitwise_xor(x, y).sum()


def tanimoto(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return -np.log2(np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum()))


def substructure(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return 1 - np.double(np.bitwise_and(x, y).sum()) / np.count_nonzero(y)


def superstructure(x, y):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return 1 - np.double(np.bitwise_and(x, y).sum()) / np.count_nonzero(x)


def get_milvus(host, port, uri=None, handler=None, **kwargs):
    if handler is None:
        handler = "GRPC"
    try_connect = kwargs.get("try_connect", True)
    if uri is not None:
        milvus = Milvus(uri=uri, handler=handler, try_connect=try_connect)
    else:
        milvus = Milvus(host=host, port=port, handler=handler, try_connect=try_connect)
    return milvus


def disable_flush(connect):
    connect.set_config("storage", "auto_flush_interval", big_flush_interval)


def enable_flush(connect):
    # reset auto_flush_interval=1
    connect.set_config("storage", "auto_flush_interval", default_flush_interval)
    config_value = connect.get_config("storage", "auto_flush_interval")
    assert config_value == str(default_flush_interval)


def gen_inaccuracy(num):
    return num / 255.0


def gen_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


# def gen_vectors(num, dim, seed=np.random.RandomState(1234), is_normal=False):
#     xb = seed.rand(num, dim).astype("float32")
#     xb = preprocessing.normalize(xb, axis=1, norm='l2')
#     return xb.tolist()


def gen_binary_vectors(num, dim):
    raw_vectors = []
    binary_vectors = []
    for i in range(num):
        raw_vector = [random.randint(0, 1) for i in range(dim)]
        raw_vectors.append(raw_vector)
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def gen_binary_sub_vectors(vectors, length):
    raw_vectors = []
    binary_vectors = []
    dim = len(vectors[0])
    for i in range(length):
        raw_vector = [0 for i in range(dim)]
        vector = vectors[i]
        for index, j in enumerate(vector):
            if j == 1:
                raw_vector[index] = 1
        raw_vectors.append(raw_vector)
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def gen_binary_super_vectors(vectors, length):
    raw_vectors = []
    binary_vectors = []
    dim = len(vectors[0])
    for i in range(length):
        cnt_1 = np.count_nonzero(vectors[i])
        raw_vector = [1 for i in range(dim)]
        raw_vectors.append(raw_vector)
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def gen_int_attr(row_num):
    return [random.randint(0, 255) for _ in range(row_num)]


def gen_float_attr(row_num):
    return [random.uniform(0, 255) for _ in range(row_num)]


def gen_unique_str(str_value=None):
    prefix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    return "test_" + prefix if str_value is None else str_value + "_" + prefix


def gen_single_filter_fields():
    fields = []
    for data_type in DataType:
        if data_type in [DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE]:
            fields.append({"field": data_type.name, "type": data_type})
    return fields


def gen_single_vector_fields():
    fields = []
    for data_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
        field = {"field": data_type.name, "type": data_type, "params": {"dim": dimension}}
        fields.append(field)
    return fields


def gen_default_fields(auto_id=False):
    default_fields = {
        "fields": [
            {"field": "int64", "type": DataType.INT64},
            {"field": "float", "type": DataType.FLOAT},
            {"field": default_float_vec_field_name, "type": DataType.FLOAT_VECTOR, "params": {"dim": dimension}},
        ],
        "segment_row_count": segment_row_count,
        "auto_id" : True
    }
    if auto_id is True:
        default_fields["auto_id"] = True
    return default_fields


def gen_binary_default_fields(auto_id=False):
    default_fields = {
        "fields": [
            {"field": "int64", "type": DataType.INT64},
            {"field": "float", "type": DataType.FLOAT},
            {"field": default_binary_vec_field_name, "type": DataType.BINARY_VECTOR, "params": {"dim": dimension}}
        ],
        "segment_row_count": segment_row_count
    }
    if auto_id is True:
        default_fields["auto_id"] = True
    return default_fields


def gen_entities(nb, is_normal=False):
    vectors = gen_vectors(nb, dimension, is_normal)
    entities = [
        {"field": "int64", "type": DataType.INT64, "values": [i for i in range(nb)]},
        {"field": "float", "type": DataType.FLOAT, "values": [float(i) for i in range(nb)]},
        {"field": default_float_vec_field_name, "type": DataType.FLOAT_VECTOR, "values": vectors}
    ]
    return entities


def gen_binary_entities(nb):
    raw_vectors, vectors = gen_binary_vectors(nb, dimension)
    entities = [
        {"field": "int64", "type": DataType.INT64, "values": [i for i in range(nb)]},
        {"field": "float", "type": DataType.FLOAT, "values": [float(i) for i in range(nb)]},
        {"field": default_binary_vec_field_name, "type": DataType.BINARY_VECTOR, "values": vectors}
    ]
    return raw_vectors, entities


def gen_entities_by_fields(fields, nb, dimension):
    entities = []
    for field in fields:
        if field["type"] in [DataType.INT32, DataType.INT64]:
            field_value = [1 for i in range(nb)]
        elif field["type"] in [DataType.FLOAT, DataType.DOUBLE]:
            field_value = [3.0 for i in range(nb)]
        elif field["type"] == DataType.BINARY_VECTOR:
            field_value = gen_binary_vectors(nb, dimension)[1]
        elif field["type"] == DataType.FLOAT_VECTOR:
            field_value = gen_vectors(nb, dimension)
        field.update({"values": field_value})
        entities.append(field)
    return entities


def assert_equal_entity(a, b):
    pass


def gen_query_vectors(field_name, entities, top_k, nq, search_params={"nprobe": 10}, rand_vector=False,
                      metric_type=None):
    if rand_vector is True:
        dimension = len(entities[-1]["values"][0])
        query_vectors = gen_vectors(nq, dimension)
    else:
        query_vectors = entities[-1]["values"][:nq]
    must_param = {"vector": {field_name: {"topk": top_k, "query": query_vectors, "params": search_params}}}
    if metric_type is not None:
        must_param["vector"][field_name]["metric_type"] = metric_type
    query = {
        "bool": {
            "must": [must_param]
        }
    }
    return query, query_vectors


def update_query_expr(src_query, keep_old=True, expr=None):
    tmp_query = copy.deepcopy(src_query)
    if expr is not None:
        tmp_query["bool"].update(expr)
    if keep_old is not True:
        tmp_query["bool"].pop("must")
    return tmp_query


def gen_default_vector_expr(default_query):
    return default_query["bool"]["must"][0]


def gen_default_term_expr(keyword="term", values=None):
    if values is None:
        values = [i for i in range(nb // 2)]
    expr = {keyword: {"int64": {"values": values}}}
    return expr


def gen_default_range_expr(ranges=None):
    if ranges is None:
        ranges = {"GT": 1, "LT": nb // 2}
    expr = {"range": {"int64": {"ranges": ranges}}}
    return expr


def add_field_default(default_fields, type=DataType.INT64, field_name=None):
    tmp_fields = copy.deepcopy(default_fields)
    if field_name is None:
        field_name = gen_unique_str()
    field = {
        "field": field_name,
        "type": type
    }
    tmp_fields["fields"].append(field)
    return tmp_fields


def add_field(entities, field_name=None):
    nb = len(entities[0]["values"])
    tmp_entities = copy.deepcopy(entities)
    if field_name is None:
        field_name = gen_unique_str()
    field = {
        "field": field_name,
        "type": DataType.INT64,
        "values": [i for i in range(nb)]
    }
    tmp_entities.append(field)
    return tmp_entities


def add_vector_field(entities, is_normal=False):
    nb = len(entities[0]["values"])
    vectors = gen_vectors(nb, dimension, is_normal)
    field = {
        "field": gen_unique_str(),
        "type": DataType.FLOAT_VECTOR,
        "values": vectors
    }
    entities.append(field)
    return entities


# def update_fields_metric_type(fields, metric_type):
#     tmp_fields = copy.deepcopy(fields)
#     if metric_type in ["L2", "IP"]:
#         tmp_fields["fields"][-1]["type"] = DataType.FLOAT_VECTOR
#     else:
#         tmp_fields["fields"][-1]["type"] = DataType.BINARY_VECTOR
#     tmp_fields["fields"][-1]["params"]["metric_type"] = metric_type
#     return tmp_fields


def remove_field(entities):
    del entities[0]
    return entities


def remove_vector_field(entities):
    del entities[-1]
    return entities


def update_field_name(entities, old_name, new_name):
    for item in entities:
        if item["field"] == old_name:
            item["field"] = new_name
    return entities


def update_field_type(entities, old_name, new_name):
    for item in entities:
        if item["field"] == old_name:
            item["type"] = new_name
    return entities


def update_field_value(entities, old_type, new_value):
    for item in entities:
        if item["type"] == old_type:
            for i in item["values"]:
                item["values"][i] = new_value
    return entities


def add_vector_field(nb, dimension=dimension):
    field_name = gen_unique_str()
    field = {
        "field": field_name,
        "type": DataType.FLOAT_VECTOR,
        "values": gen_vectors(nb, dimension)
    }
    return field_name


def gen_segment_row_counts():
    sizes = [
        1,
        2,
        1024,
        4096
    ]
    return sizes


def gen_invalid_ips():
    ips = [
        # "255.0.0.0",
        # "255.255.0.0",
        # "255.255.255.0",
        # "255.255.255.255",
        "127.0.0",
        # "123.0.0.2",
        "12-s",
        " ",
        "12 s",
        "BB。A",
        " siede ",
        "(mn)",
        "中文",
        "a".join("a" for _ in range(256))
    ]
    return ips


def gen_invalid_uris():
    ip = None
    uris = [
        " ",
        "中文",
        # invalid protocol
        # "tc://%s:%s" % (ip, port),
        # "tcp%s:%s" % (ip, port),

        # # invalid port
        # "tcp://%s:100000" % ip,
        # "tcp://%s: " % ip,
        # "tcp://%s:19540" % ip,
        # "tcp://%s:-1" % ip,
        # "tcp://%s:string" % ip,

        # invalid ip
        "tcp:// :19530",
        # "tcp://123.0.0.1:%s" % port,
        "tcp://127.0.0:19530",
        # "tcp://255.0.0.0:%s" % port,
        # "tcp://255.255.0.0:%s" % port,
        # "tcp://255.255.255.0:%s" % port,
        # "tcp://255.255.255.255:%s" % port,
        "tcp://\n:19530",
    ]
    return uris


def gen_invalid_strs():
    strings = [
        1,
        [1],
        None,
        "12-s",
        " ",
        # "",
        # None,
        "12 s",
        "BB。A",
        "c|c",
        " siede ",
        "(mn)",
        "pip+",
        "=c",
        "中文",
        "a".join("a" for i in range(256))
    ]
    return strings


def gen_invalid_field_types():
    field_types = [
        # 1,
        "=c",
        # 0,
        None,
        "",
        "a".join("a" for i in range(256))
    ]
    return field_types


def gen_invalid_metric_types():
    metric_types = [
        1,
        "=c",
        0,
        None,
        "",
        "a".join("a" for i in range(256))
    ]
    return metric_types


# TODO:
def gen_invalid_ints():
    top_ks = [
        # 1.0,
        None,
        "stringg",
        [1, 2, 3],
        (1, 2),
        {"a": 1},
        " ",
        "",
        "String",
        "12-s",
        "BB。A",
        " siede ",
        "(mn)",
        "pip+",
        "=c",
        "中文",
        "a".join("a" for i in range(256))
    ]
    return top_ks


def gen_invalid_params():
    params = [
        9999999999,
        -1,
        # None,
        [1, 2, 3],
        (1, 2),
        {"a": 1},
        " ",
        "",
        "String",
        "12-s",
        "BB。A",
        " siede ",
        "(mn)",
        "pip+",
        "=c",
        "中文"
    ]
    return params


def gen_invalid_vectors():
    invalid_vectors = [
        "1*2",
        [],
        [1],
        [1, 2],
        [" "],
        ['a'],
        [None],
        None,
        (1, 2),
        {"a": 1},
        " ",
        "",
        "String",
        "12-s",
        "BB。A",
        " siede ",
        "(mn)",
        "pip+",
        "=c",
        "中文",
        "a".join("a" for i in range(256))
    ]
    return invalid_vectors


def gen_invaild_search_params():
    invalid_search_key = 100
    search_params = []
    for index_type in all_index_types:
        if index_type == "FLAT":
            continue
        search_params.append({"index_type": index_type, "search_params": {"invalid_key": invalid_search_key}})
        if index_type in delete_support():
            for nprobe in gen_invalid_params():
                ivf_search_params = {"index_type": index_type, "search_params": {"nprobe": nprobe}}
                search_params.append(ivf_search_params)
        elif index_type == "HNSW":
            for ef in gen_invalid_params():
                hnsw_search_param = {"index_type": index_type, "search_params": {"ef": ef}}
                search_params.append(hnsw_search_param)
        elif index_type == "NSG":
            for search_length in gen_invalid_params():
                nsg_search_param = {"index_type": index_type, "search_params": {"search_length": search_length}}
                search_params.append(nsg_search_param)
            search_params.append({"index_type": index_type, "search_params": {"invalid_key": 100}})
        elif index_type == "ANNOY":
            for search_k in gen_invalid_params():
                if isinstance(search_k, int):
                    continue
                annoy_search_param = {"index_type": index_type, "search_params": {"search_k": search_k}}
                search_params.append(annoy_search_param)
    return search_params


def gen_invalid_index():
    index_params = []
    for index_type in gen_invalid_strs():
        index_param = {"index_type": index_type, "params": {"nlist": 1024}}
        index_params.append(index_param)
    for nlist in gen_invalid_params():
        index_param = {"index_type": "IVF_FLAT", "params": {"nlist": nlist}}
        index_params.append(index_param)
    for M in gen_invalid_params():
        index_param = {"index_type": "HNSW", "params": {"M": M, "efConstruction": 100}}
        index_params.append(index_param)
    for efConstruction in gen_invalid_params():
        index_param = {"index_type": "HNSW", "params": {"M": 16, "efConstruction": efConstruction}}
        index_params.append(index_param)
    for search_length in gen_invalid_params():
        index_param = {"index_type": "NSG",
                       "params": {"search_length": search_length, "out_degree": 40, "candidate_pool_size": 50,
                                  "knng": 100}}
        index_params.append(index_param)
    for out_degree in gen_invalid_params():
        index_param = {"index_type": "NSG",
                       "params": {"search_length": 100, "out_degree": out_degree, "candidate_pool_size": 50,
                                  "knng": 100}}
        index_params.append(index_param)
    for candidate_pool_size in gen_invalid_params():
        index_param = {"index_type": "NSG", "params": {"search_length": 100, "out_degree": 40,
                                                       "candidate_pool_size": candidate_pool_size,
                                                       "knng": 100}}
        index_params.append(index_param)
    index_params.append({"index_type": "IVF_FLAT", "params": {"invalid_key": 1024}})
    index_params.append({"index_type": "HNSW", "params": {"invalid_key": 16, "efConstruction": 100}})
    index_params.append({"index_type": "NSG",
                         "params": {"invalid_key": 100, "out_degree": 40, "candidate_pool_size": 300,
                                    "knng": 100}})
    for invalid_n_trees in gen_invalid_params():
        index_params.append({"index_type": "ANNOY", "params": {"n_trees": invalid_n_trees}})

    return index_params


def gen_index():
    nlists = [1, 1024, 16384]
    pq_ms = [128, 64, 32, 16, 8, 4]
    Ms = [5, 24, 48]
    efConstructions = [100, 300, 500]
    search_lengths = [10, 100, 300]
    out_degrees = [5, 40, 300]
    candidate_pool_sizes = [50, 100, 300]
    knngs = [5, 100, 300]

    index_params = []
    for index_type in all_index_types:
        if index_type in ["FLAT", "BIN_FLAT", "BIN_IVF_FLAT"]:
            index_params.append({"index_type": index_type, "index_param": {"nlist": 1024}})
        elif index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_SQ8_HYBRID"]:
            ivf_params = [{"index_type": index_type, "index_param": {"nlist": nlist}} \
                          for nlist in nlists]
            index_params.extend(ivf_params)
        elif index_type == "IVF_PQ":
            IVFPQ_params = [{"index_type": index_type, "index_param": {"nlist": nlist, "m": m}} \
                            for nlist in nlists \
                            for m in pq_ms]
            index_params.extend(IVFPQ_params)
        elif index_type == "HNSW":
            hnsw_params = [{"index_type": index_type, "index_param": {"M": M, "efConstruction": efConstruction}} \
                           for M in Ms \
                           for efConstruction in efConstructions]
            index_params.extend(hnsw_params)
        elif index_type == "NSG":
            nsg_params = [{"index_type": index_type,
                           "index_param": {"search_length": search_length, "out_degree": out_degree,
                                           "candidate_pool_size": candidate_pool_size, "knng": knng}} \
                          for search_length in search_lengths \
                          for out_degree in out_degrees \
                          for candidate_pool_size in candidate_pool_sizes \
                          for knng in knngs]
            index_params.extend(nsg_params)

    return index_params


def gen_simple_index():
    index_params = []
    for i in range(len(all_index_types)):
        if all_index_types[i] in binary_support():
            continue
        dic = {"index_type": all_index_types[i], "metric_type": "L2"}
        dic.update({"params": default_index_params[i]})
        index_params.append(dic)
    return index_params


def gen_binary_index():
    index_params = []
    for i in range(len(all_index_types)):
        if all_index_types[i] in binary_support():
            dic = {"index_type": all_index_types[i]}
            dic.update({"params": default_index_params[i]})
            index_params.append(dic)
    return index_params


def get_search_param(index_type):
    search_params = {"metric_type": "L2"}
    if index_type in ivf() or index_type in binary_support():
        search_params.update({"nprobe": 32})
    elif index_type == "HNSW":
        search_params.update({"ef": 64})
    elif index_type == "NSG":
        search_params.update({"search_length": 100})
    elif index_type == "ANNOY":
        search_params.update({"search_k": 100})
    else:
        logging.getLogger().error("Invalid index_type.")
        raise Exception("Invalid index_type.")
    return search_params


def assert_equal_vector(v1, v2):
    if len(v1) != len(v2):
        assert False
    for i in range(len(v1)):
        assert abs(v1[i] - v2[i]) < epsilon


def restart_server(helm_release_name):
    res = True
    timeout = 120
    from kubernetes import client, config
    client.rest.logger.setLevel(logging.WARNING)

    namespace = "milvus"
    # service_name = "%s.%s.svc.cluster.local" % (helm_release_name, namespace)
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pod_name = None
    # config_map_names = v1.list_namespaced_config_map(namespace, pretty='true')
    # body = {"replicas": 0}
    pods = v1.list_namespaced_pod(namespace)
    for i in pods.items:
        if i.metadata.name.find(helm_release_name) != -1 and i.metadata.name.find("mysql") == -1:
            pod_name = i.metadata.name
            break
            # v1.patch_namespaced_config_map(config_map_name, namespace, body, pretty='true')
    # status_res = v1.read_namespaced_service_status(helm_release_name, namespace, pretty='true')
    # print(status_res)
    if pod_name is not None:
        try:
            v1.delete_namespaced_pod(pod_name, namespace)
        except Exception as e:
            logging.error(str(e))
            logging.error("Exception when calling CoreV1Api->delete_namespaced_pod")
            res = False
            return res
        time.sleep(5)
        # check if restart successfully
        pods = v1.list_namespaced_pod(namespace)
        for i in pods.items:
            pod_name_tmp = i.metadata.name
            if pod_name_tmp.find(helm_release_name) != -1:
                logging.debug(pod_name_tmp)
                start_time = time.time()
                while time.time() - start_time > timeout:
                    status_res = v1.read_namespaced_pod_status(pod_name_tmp, namespace, pretty='true')
                    if status_res.status.phase == "Running":
                        break
                    time.sleep(1)
                if time.time() - start_time > timeout:
                    logging.error("Restart pod: %s timeout" % pod_name_tmp)
                    res = False
                    return res
    else:
        logging.error("Pod: %s not found" % helm_release_name)
        res = False
    return res

// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "server/ValidationUtil.h"
#include "db/Utils.h"
#include "knowhere/index/vector_index/ConfAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "utils/Log.h"
#include "utils/StringHelpFunctions.h"

#include <fiu-local.h>
#include <limits>
#include <set>
#include <string>

namespace milvus {
namespace server {

namespace {

Status
CheckParameterRange(const milvus::json& json_params, const std::string& param_name, int64_t min, int64_t max,
                    bool min_close = true, bool max_closed = true) {
    if (json_params.find(param_name) == json_params.end()) {
        std::string msg = "Parameter list must contain: ";
        msg += param_name;
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    try {
        int64_t value = json_params[param_name];
        bool min_err = min_close ? value < min : value <= min;
        bool max_err = max_closed ? value > max : value >= max;
        if (min_err || max_err) {
            std::string msg = "Invalid " + param_name + " value: " + std::to_string(value) + ". Valid range is " +
                              (min_close ? "[" : "(") + std::to_string(min) + ", " + std::to_string(max) +
                              (max_closed ? "]" : ")");
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    } catch (std::exception& e) {
        std::string msg = "Invalid " + param_name + ": ";
        msg += e.what();
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}

Status
CheckParameterExistence(const milvus::json& json_params, const std::string& param_name) {
    if (json_params.find(param_name) == json_params.end()) {
        std::string msg = "Parameter list must contain: ";
        msg += param_name;
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    try {
        int64_t value = json_params[param_name];
        if (value < 0) {
            std::string msg = "Invalid " + param_name + " value: " + std::to_string(value);
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    } catch (std::exception& e) {
        std::string msg = "Invalid " + param_name + ": ";
        msg += e.what();
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_ARGUMENT, msg);
    }

    return Status::OK();
}

}  // namespace

Status
ValidateCollectionName(const std::string& collection_name) {
    // Collection name shouldn't be empty.
    if (collection_name.empty()) {
        std::string msg = "Collection name should not be empty.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    std::string invalid_msg = "Invalid collection name: " + collection_name + ". ";
    // Collection name size shouldn't exceed 255.
    if (collection_name.size() > engine::MAX_NAME_LENGTH) {
        std::string msg = invalid_msg + "The length of a collection name must be less than 255 characters.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    // Collection name first character should be underscore or character.
    char first_char = collection_name[0];
    if (first_char != '_' && std::isalpha(first_char) == 0) {
        std::string msg = invalid_msg + "The first character of a collection name must be an underscore or letter.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    int64_t table_name_size = collection_name.size();
    for (int64_t i = 1; i < table_name_size; ++i) {
        char name_char = collection_name[i];
        if (name_char != '_' && std::isalnum(name_char) == 0) {
            std::string msg = invalid_msg + "Collection name can only contain numbers, letters, and underscores.";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_COLLECTION_NAME, msg);
        }
    }

    return Status::OK();
}

Status
ValidateFieldName(const std::string& field_name) {
    // Field name shouldn't be empty.
    if (field_name.empty()) {
        std::string msg = "Field name should not be empty.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_FIELD_NAME, msg);
    }

    std::string invalid_msg = "Invalid field name: " + field_name + ". ";
    // Field name size shouldn't exceed 255.
    if (field_name.size() > engine::MAX_NAME_LENGTH) {
        std::string msg = invalid_msg + "The length of a field name must be less than 255 characters.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_FIELD_NAME, msg);
    }

    // Field name first character should be underscore or character.
    char first_char = field_name[0];
    if (first_char != '_' && std::isalpha(first_char) == 0) {
        std::string msg = invalid_msg + "The first character of a field name must be an underscore or letter.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_FIELD_NAME, msg);
    }

    int64_t field_name_size = field_name.size();
    for (int64_t i = 1; i < field_name_size; ++i) {
        char name_char = field_name[i];
        if (name_char != '_' && std::isalnum(name_char) == 0) {
            std::string msg = invalid_msg + "Field name cannot only contain numbers, letters, and underscores.";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_FIELD_NAME, msg);
        }
    }

    return Status::OK();
}

Status
ValidateIndexType(const std::string& index_type) {
    // Index name shouldn't be empty.
    if (index_type.empty()) {
        std::string msg = "Index type should not be empty.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_FIELD_NAME, msg);
    }

    static std::set<std::string> s_valid_index_names = {
        knowhere::IndexEnum::INVALID,
        knowhere::IndexEnum::INDEX_FAISS_IDMAP,
        knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
        knowhere::IndexEnum::INDEX_FAISS_IVFPQ,
        knowhere::IndexEnum::INDEX_FAISS_IVFSQ8,
        knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H,
        knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP,
        knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT,
        knowhere::IndexEnum::INDEX_NSG,
        knowhere::IndexEnum::INDEX_HNSW,
        knowhere::IndexEnum::INDEX_ANNOY,
        knowhere::IndexEnum::INDEX_HNSW_SQ8NM,
    };

    if (s_valid_index_names.find(index_type) == s_valid_index_names.end()) {
        std::string msg = "Invalid index type: " + index_type;
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_INDEX_TYPE, msg);
    }

    return Status::OK();
}

Status
ValidateDimension(int64_t dim, bool is_binary) {
    if (dim <= 0 || dim > engine::MAX_DIMENSION) {
        std::string msg = "Invalid dimension: " + std::to_string(dim) + ". Should be in range 1 ~ " +
                          std::to_string(engine::MAX_DIMENSION) + ".";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_VECTOR_DIMENSION, msg);
    }

    if (is_binary && (dim % 8) != 0) {
        std::string msg = "Invalid dimension: " + std::to_string(dim) + ". Should be multiple of 8.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_VECTOR_DIMENSION, msg);
    }

    return Status::OK();
}

Status
ValidateIndexParams(const milvus::json& index_params, int64_t dimension, const std::string& index_type) {
    if (index_type == knowhere::IndexEnum::INDEX_FAISS_IDMAP ||
        index_type == knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP) {
        return Status::OK();
    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT ||
               index_type == knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 ||
               index_type == knowhere::IndexEnum::INDEX_FAISS_IVFSQ8H ||
               index_type == knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
        auto status = CheckParameterRange(index_params, knowhere::IndexParams::nlist, 1, 999999);
        if (!status.ok()) {
            return status;
        }
    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
        auto status = CheckParameterRange(index_params, knowhere::IndexParams::nlist, 1, 999999);
        if (!status.ok()) {
            return status;
        }

        status = CheckParameterExistence(index_params, knowhere::IndexParams::m);
        if (!status.ok()) {
            return status;
        }

        // special check for 'm' parameter
        std::vector<int64_t> resset;
        milvus::knowhere::IVFPQConfAdapter::GetValidMList(dimension, resset);
        int64_t m_value = index_params[knowhere::IndexParams::m];
        if (resset.empty()) {
            std::string msg = "Invalid collection dimension, unable to get reasonable values for 'm'";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_COLLECTION_DIMENSION, msg);
        }

        auto iter = std::find(std::begin(resset), std::end(resset), m_value);
        if (iter == std::end(resset)) {
            std::string msg =
                "Invalid " + std::string(knowhere::IndexParams::m) + ", must be one of the following values: ";
            for (size_t i = 0; i < resset.size(); i++) {
                if (i != 0) {
                    msg += ",";
                }
                msg += std::to_string(resset[i]);
            }

            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_ARGUMENT, msg);
        }
    } else if (index_type == knowhere::IndexEnum::INDEX_NSG) {
        auto status = CheckParameterRange(index_params, knowhere::IndexParams::search_length, 10, 300);
        if (!status.ok()) {
            return status;
        }
        status = CheckParameterRange(index_params, knowhere::IndexParams::out_degree, 5, 300);
        if (!status.ok()) {
            return status;
        }
        status = CheckParameterRange(index_params, knowhere::IndexParams::candidate, 50, 1000);
        if (!status.ok()) {
            return status;
        }
        status = CheckParameterRange(index_params, knowhere::IndexParams::knng, 5, 300);
        if (!status.ok()) {
            return status;
        }
    } else if (index_type == knowhere::IndexEnum::INDEX_HNSW || index_type == knowhere::IndexEnum::INDEX_HNSW_SQ8NM) {
        auto status = CheckParameterRange(index_params, knowhere::IndexParams::M, 4, 64);
        if (!status.ok()) {
            return status;
        }
        status = CheckParameterRange(index_params, knowhere::IndexParams::efConstruction, 8, 512);
        if (!status.ok()) {
            return status;
        }
    } else if (index_type == knowhere::IndexEnum::INDEX_ANNOY) {
        auto status = CheckParameterRange(index_params, knowhere::IndexParams::n_trees, 1, 1024);
        if (!status.ok()) {
            return status;
        }
    }

    return Status::OK();
}

Status
ValidateSegmentRowCount(int64_t segment_row_count) {
    if (segment_row_count <= 0 || segment_row_count > engine::MAX_SEGMENT_ROW_COUNT) {
        std::string msg = "Invalid segment row count: " + std::to_string(segment_row_count) + ". " +
                          "Should be in range 1 ~ " + std::to_string(engine::MAX_SEGMENT_ROW_COUNT) + ".";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_SEGMENT_ROW_COUNT, msg);
    }
    return Status::OK();
}

Status
ValidateIndexMetricType(const std::string& metric_type) {
    static std::set<std::string> s_valid_metric = {
        milvus::knowhere::Metric::L2,
        milvus::knowhere::Metric::IP,
        milvus::knowhere::Metric::HAMMING,
        milvus::knowhere::Metric::JACCARD,
        milvus::knowhere::Metric::TANIMOTO,
        milvus::knowhere::Metric::SUBSTRUCTURE,
        milvus::knowhere::Metric::SUPERSTRUCTURE,
    };
    if (s_valid_metric.find(metric_type) == s_valid_metric.end()) {
        std::string msg =
            "Invalid index metric type: " + metric_type + ". " + "Make sure the metric type is in MetricType list.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_INDEX_METRIC_TYPE, msg);
    }
    return Status::OK();
}

Status
ValidateSearchTopk(int64_t top_k) {
    if (top_k <= 0 || top_k > QUERY_MAX_TOPK) {
        std::string msg =
            "Invalid topk: " + std::to_string(top_k) + ". " + "The topk must be within the range of 1 ~ 2048.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_TOPK, msg);
    }

    return Status::OK();
}

Status
ValidatePartitionName(const std::string& partition_name) {
    if (partition_name.empty()) {
        std::string msg = "Partition name should not be empty.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    std::string invalid_msg = "Invalid partition name: " + partition_name + ". ";
    // Collection name size shouldn't exceed 255.
    if (partition_name.size() > engine::MAX_NAME_LENGTH) {
        std::string msg = invalid_msg + "The length of a partition name must be less than 255 characters.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    // Collection name first character should be underscore or character.
    char first_char = partition_name[0];
    if (first_char != '_' && std::isalpha(first_char) == 0) {
        std::string msg = invalid_msg + "The first character of a partition name must be an underscore or letter.";
        LOG_SERVER_ERROR_ << msg;
        return Status(SERVER_INVALID_COLLECTION_NAME, msg);
    }

    int64_t table_name_size = partition_name.size();
    for (int64_t i = 1; i < table_name_size; ++i) {
        char name_char = partition_name[i];
        if (name_char != '_' && std::isalnum(name_char) == 0) {
            std::string msg = invalid_msg + "Partition name can only contain numbers, letters, and underscores.";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_COLLECTION_NAME, msg);
        }
    }

    return Status::OK();
}

Status
ValidatePartitionTags(const std::vector<std::string>& partition_tags) {
    for (const std::string& tag : partition_tags) {
        // trim side-blank of tag, only compare valid characters
        // for example: " ab cd " is treated as "ab cd"
        std::string valid_tag = tag;
        StringHelpFunctions::TrimStringBlank(valid_tag);
        if (valid_tag.empty()) {
            std::string msg = "Invalid partition tag: " + valid_tag + ". " + "Partition tag should not be empty.";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_PARTITION_TAG, msg);
        }

        // max length of partition tag
        if (valid_tag.length() > 255) {
            std::string msg = "Invalid partition tag: " + valid_tag + ". " + "Partition tag exceed max length(255).";
            LOG_SERVER_ERROR_ << msg;
            return Status(SERVER_INVALID_PARTITION_TAG, msg);
        }
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus

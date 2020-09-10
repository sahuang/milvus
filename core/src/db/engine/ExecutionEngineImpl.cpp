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

#include "db/engine/ExecutionEngineImpl.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "config/ServerConfig.h"
#include "db/SnapshotUtils.h"
#include "db/Utils.h"
#include "segment/SegmentReader.h"
#include "segment/SegmentWriter.h"
#include "utils/CommonUtil.h"
#include "utils/Error.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/Status.h"
#include "utils/TimeRecorder.h"

#include "knowhere/common/Config.h"
#include "knowhere/index/structured_index/StructuredIndexSort.h"
#include "knowhere/index/vector_index/ConfAdapter.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexBinaryIDMAP.h"
#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/VecIndexFactory.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

#ifdef MILVUS_GPU_VERSION
#include "knowhere/index/vector_index/gpu/GPUIndex.h"
#include "knowhere/index/vector_index/gpu/IndexIVFSQHybrid.h"
#include "knowhere/index/vector_index/gpu/Quantizer.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#endif

namespace milvus {
namespace engine {
namespace {
template <typename T>
knowhere::IndexPtr
CreateSortedIndex(engine::BinaryDataPtr& raw_data) {
    if (raw_data == nullptr) {
        return nullptr;
    }

    auto count = raw_data->data_.size() / sizeof(T);
    auto index_ptr =
        std::make_shared<knowhere::StructuredIndexSort<T>>(count, reinterpret_cast<const T*>(raw_data->data_.data()));
    return std::static_pointer_cast<knowhere::Index>(index_ptr);
}
}  // namespace

ExecutionEngineImpl::ExecutionEngineImpl(const std::string& dir_root, const SegmentVisitorPtr& segment_visitor)
    : gpu_enable_(config.gpu.enable()) {
    segment_reader_ = std::make_shared<segment::SegmentReader>(dir_root, segment_visitor);
}

knowhere::VecIndexPtr
ExecutionEngineImpl::CreateVecIndex(const std::string& index_name) {
    knowhere::VecIndexFactory& vec_index_factory = knowhere::VecIndexFactory::GetInstance();
    knowhere::IndexMode mode = knowhere::IndexMode::MODE_CPU;
#ifdef MILVUS_GPU_VERSION
    if (gpu_enable_) {
        mode = knowhere::IndexMode::MODE_GPU;
    }
#endif

    knowhere::VecIndexPtr index = vec_index_factory.CreateVecIndex(index_name, mode);
    if (index == nullptr) {
        std::string err_msg =
            "Invalid index type: " + index_name + " mode: " + std::to_string(static_cast<int32_t>(mode));
        LOG_ENGINE_ERROR_ << err_msg;
    }
    return index;
}

double
ExecutionEngineImpl::getmillisecs () {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

Status
ExecutionEngineImpl::Load(ExecutionEngineContext& context) {
    if (context.query_ptr_ != nullptr) {
        context_ = context;
        return LoadForSearch(context.query_ptr_);
    } else {
        return Load(context.target_fields_);
    }
}

Status
ExecutionEngineImpl::LoadForSearch(const query::QueryPtr& query_ptr) {
    return Load(query_ptr->index_fields);
}

Status
ExecutionEngineImpl::CreateStructuredIndex(const DataType field_type, engine::BinaryDataPtr& raw_data,
                                           knowhere::IndexPtr& index_ptr) {
    switch (field_type) {
        case engine::DataType::INT32: {
            index_ptr = CreateSortedIndex<int32_t>(raw_data);
            break;
        }
        case engine::DataType::INT64: {
            index_ptr = CreateSortedIndex<int64_t>(raw_data);
            break;
        }
        case engine::DataType::FLOAT: {
            index_ptr = CreateSortedIndex<float>(raw_data);
            break;
        }
        case engine::DataType::DOUBLE: {
            index_ptr = CreateSortedIndex<double>(raw_data);
            break;
        }
        default: { return Status(DB_ERROR, "Field is not structured type"); }
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::Load(const TargetFields& field_names) {
    TimeRecorderAuto rc("ExecutionEngineImpl::Load");

    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);
    auto segment_visitor = segment_reader_->GetSegmentVisitor();

    for (auto& name : field_names) {
        DataType field_type = DataType::NONE;
        segment_ptr->GetFieldType(name, field_type);

        bool index_exist = false;
        if (field_type == DataType::VECTOR_FLOAT || field_type == DataType::VECTOR_BINARY) {
            bool valid_metric_type = false;
            if (!context_.query_ptr_) {
                valid_metric_type = true;
            } else {
                auto field_visitor = segment_visitor->GetFieldVisitor(name);
                auto field_element_visitor = field_visitor->GetElementVisitor(engine::FieldElementType::FET_INDEX);
                if (field_element_visitor) {
                    auto field_element = field_element_visitor->GetElement();
                    if (field_element->GetParams().contains(engine::PARAM_INDEX_METRIC_TYPE)) {
                        std::string metric_type = field_element->GetParams()[engine::PARAM_INDEX_METRIC_TYPE];
                        if (context_.query_ptr_->metric_types.find(name) == context_.query_ptr_->metric_types.end()) {
                            valid_metric_type = true;
                        } else if (context_.query_ptr_->metric_types.at(name) == metric_type) {
                            valid_metric_type = true;
                        }
                    }
                }
                //                else {
                //                    if (context_->query_ptr_->metric_types.find(name) ==
                //                    context_->query_ptr_->metric_types.end()) {
                //                        return Status{DB_ERROR,
                //                                      "Please provide a metric_type in search params since index is
                //                                      not created"};
                //                    }
                //                }
            }

            knowhere::VecIndexPtr index_ptr;
            if (valid_metric_type) {
                segment_reader_->LoadVectorIndex(name, index_ptr);
                index_exist = (index_ptr != nullptr);
            } else {
                segment_reader_->LoadVectorIndex(name, index_ptr, true);
                index_exist = (index_ptr != nullptr);
            }
            knowhere::VecIndexPtr flat_index;
            segment_reader_->LoadFlatIndex(name, flat_index, "_flat");
            for (int64_t i = 0; i < index_ptr->GetUids().size(); i++) {
                uid2off_.insert({index_ptr->GetUids()[i], i});
            }
        } else {
            knowhere::IndexPtr index_ptr;
            STATUS_CHECK(segment_reader_->LoadStructuredIndex(name, index_ptr));
            index_exist = (index_ptr != nullptr);
            if (!index_exist) {
                // for structured field, create a simple sorted index for it
                // we also can do this in BuildIndex step, but for now we do this in Load step
                BinaryDataPtr raw_data;
                segment_reader_->LoadField(name, raw_data);
                STATUS_CHECK(CreateStructuredIndex(field_type, raw_data, index_ptr));
                segment_ptr->SetStructuredIndex(name, index_ptr);
                index_exist = true;
            }
        }

        // index not yet build, load raw data
        if (!index_exist) {
            BinaryDataPtr raw;
            segment_reader_->LoadField(name, raw);
        }

        target_fields_.insert(name);
    }

    return Status::OK();
}

Status
ExecutionEngineImpl::CopyToGpu(uint64_t device_id) {
#ifdef MILVUS_GPU_VERSION
    TimeRecorderAuto rc("ExecutionEngineImpl::CopyToGpu");

    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);

    engine::VECTOR_INDEX_MAP new_map;
    engine::VECTOR_INDEX_MAP& indice = segment_ptr->GetVectorIndice();
    for (auto& pair : indice) {
        if (pair.second != nullptr) {
            auto gpu_index = knowhere::cloner::CopyCpuToGpu(pair.second, device_id, knowhere::Config());
            new_map.insert(std::make_pair(pair.first, gpu_index));
        }
    }

    indice.swap(new_map);
    gpu_num_ = device_id;
#endif
    return Status::OK();
}

void
MapAndCopyResult(const knowhere::DatasetPtr& dataset, const std::vector<idx_t>& uids, int64_t nq, int64_t k,
                 float* distances, int64_t* labels) {
    auto res_ids = dataset->Get<int64_t*>(knowhere::meta::IDS);
    auto res_dist = dataset->Get<float*>(knowhere::meta::DISTANCE);

    memcpy(distances, res_dist, sizeof(float) * nq * k);

    /* map offsets to ids */
    int64_t num = nq * k;
    for (int64_t i = 0; i < num; ++i) {
        int64_t offset = res_ids[i];
        if (offset != -1) {
            labels[i] = uids[offset];
        } else {
            labels[i] = -1;
        }
    }

    free(res_ids);
    free(res_dist);
}

Status
ExecutionEngineImpl::VecSearch(milvus::engine::ExecutionEngineContext& context,
                               const query::VectorQueryPtr& vector_param, knowhere::VecIndexPtr& vec_index,
                               bool hybrid) {
    TimeRecorder rc(LogOut("[%s][%ld] ExecutionEngineImpl::Search", "search", 0));

    if (vec_index == nullptr) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] ExecutionEngineImpl: index is null, failed to search", "search", 0);
        return Status(DB_ERROR, "index is null");
    }

    uint64_t nq = vector_param->nq;
    auto query_vector = vector_param->query_vector;
    uint64_t topk = vector_param->topk;

    context.query_result_ = std::make_shared<QueryResult>();
    context.query_result_->result_ids_.resize(topk * nq);
    context.query_result_->result_distances_.resize(topk * nq);

    milvus::json conf = vector_param->extra_params;
    conf[knowhere::meta::TOPK] = topk;
    conf[knowhere::Metric::TYPE] = vector_param->metric_type;
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(vec_index->index_type());
    if (!adapter->CheckSearch(conf, vec_index->index_type(), vec_index->index_mode())) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] Illegal search params", "search", 0);
        throw Exception(DB_ERROR, "Illegal search params");
    }

    if (hybrid) {
        //        HybridLoad();
    }

    rc.RecordSection("query prepare");
    knowhere::DatasetPtr dataset;
    if (!query_vector.float_data.empty()) {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.float_data.data());
    } else {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.binary_data.data());
    }
    auto result = vec_index->Query(dataset, conf);

    MapAndCopyResult(result, vec_index->GetUids(), nq, topk, context.query_result_->result_distances_.data(),
                     context.query_result_->result_ids_.data());

    if (hybrid) {
        //        HybridUnset();
    }

    return Status::OK();
}

Status
ExecutionEngineImpl::VecSearchWithOptimizer(milvus::engine::ExecutionEngineContext& context,
                                            const query::VectorQueryPtr& vector_param, knowhere::VecIndexPtr& vec_index,
                                            float delta, bool expand) {
    TimeRecorder rc(LogOut("[%s][%ld] ExecutionEngineImpl::Search", "search", 0));

    if (vec_index == nullptr) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] ExecutionEngineImpl: index is null, failed to search", "search", 0);
        return Status(DB_ERROR, "index is null");
    }

    uint64_t nq = vector_param->nq;
    auto query_vector = vector_param->query_vector;
    uint64_t topk = vector_param->topk;
    // maybe need to expand topk
    if (expand)
        topk = (int64_t) (1.0 * topk * delta);

    context.query_result_ = std::make_shared<QueryResult>();
    context.query_result_->result_ids_.resize(topk * nq);
    context.query_result_->result_distances_.resize(topk * nq);

    milvus::json conf = vector_param->extra_params;
    conf[knowhere::meta::TOPK] = topk;
    conf[knowhere::Metric::TYPE] = vector_param->metric_type;
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(vec_index->index_type());
    if (!adapter->CheckSearch(conf, vec_index->index_type(), vec_index->index_mode())) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] Illegal search params", "search", 0);
        throw Exception(DB_ERROR, "Illegal search params");
    }

    rc.RecordSection("query prepare");
    knowhere::DatasetPtr dataset;
    if (!query_vector.float_data.empty()) {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.float_data.data());
    } else {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.binary_data.data());
    }
    auto result = vec_index->Query(dataset, conf);

    if (!expand)
        MapAndCopyResult(result, vec_index->GetUids(), nq, topk, context.query_result_->result_distances_.data(),
                     context.query_result_->result_ids_.data());

    return Status::OK();
}

Status
ExecutionEngineImpl::VecSearchWithFlat(ExecutionEngineContext& context, const query::VectorQueryPtr& vector_param,
                                       knowhere::VecIndexPtr& vec_index, std::vector<int64_t>& offset) {
    TimeRecorder rc(LogOut("[%s][%ld] ExecutionEngineImpl::Search", "search", 0));

    if (vec_index == nullptr) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] ExecutionEngineImpl: index is null, failed to search", "search", 0);
        return Status(DB_ERROR, "index is null");
    }

    uint64_t nq = vector_param->nq;
    auto query_vector = vector_param->query_vector;
    uint64_t topk = vector_param->topk;

    context.query_result_ = std::make_shared<QueryResult>();
    context.query_result_->result_ids_.resize(topk * nq);
    context.query_result_->result_distances_.resize(topk * nq);

    milvus::json conf = vector_param->extra_params;
    conf[knowhere::meta::TOPK] = topk;
    conf[knowhere::Metric::TYPE] = vector_param->metric_type;
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(vec_index->index_type());
    if (!adapter->CheckSearch(conf, vec_index->index_type(), vec_index->index_mode())) {
        LOG_ENGINE_ERROR_ << LogOut("[%s][%ld] Illegal search params", "search", 0);
        throw Exception(DB_ERROR, "Illegal search params");
    }

    rc.RecordSection("query prepare");
    knowhere::DatasetPtr dataset;
    if (!query_vector.float_data.empty()) {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.float_data.data());
    } else {
        dataset = knowhere::GenDataset(nq, vec_index->Dim(), query_vector.binary_data.data());
    }
    auto result = vec_index->QueryWithOffset(dataset, conf, offset);

    MapAndCopyResult(result, vec_index->GetUids(), nq, topk, context.query_result_->result_distances_.data(),
                     context.query_result_->result_ids_.data());

    return Status::OK();
}

Status
ExecutionEngineImpl::Search(ExecutionEngineContext& context) {
    try {
        faiss::ConcurrentBitsetPtr bitset;
        std::string vector_placeholder;
        faiss::ConcurrentBitsetPtr list;

        SegmentPtr segment_ptr;
        segment_reader_->GetSegment(segment_ptr);
        knowhere::VecIndexPtr vec_index = nullptr;
        std::unordered_map<std::string, engine::DataType> attr_type;

        auto segment_visitor = segment_reader_->GetSegmentVisitor();
        auto field_visitors = segment_visitor->GetFieldVisitors();
        for (const auto& name : context.query_ptr_->index_fields) {
            auto field_visitor = segment_visitor->GetFieldVisitor(name);
            if (!field_visitor) {
                return Status(SERVER_INVALID_DSL_PARAMETER, "Field: " + name + " is not existed");
            }
            auto field = field_visitor->GetField();
            if (field->GetFtype() == static_cast<snapshot::FTYPE_TYPE>(engine::DataType::VECTOR_FLOAT) ||
                field->GetFtype() == static_cast<snapshot::FTYPE_TYPE>(engine::DataType::VECTOR_BINARY)) {
                STATUS_CHECK(segment_ptr->GetVectorIndex(name, vec_index));
            } else {
                attr_type.insert(std::make_pair(name, static_cast<engine::DataType>(field->GetFtype())));
            }
        }

        list = vec_index->GetBlacklist();
        entity_count_ = list->capacity();
        // Parse general query
        auto status = ExecBinaryQuery(context.query_ptr_->root, bitset, attr_type, vector_placeholder);
        if (!status.ok()) {
            return status;
        }

        // Do And
        for (int64_t i = 0; i < entity_count_; i++) {
            if (!list->test(i) && !bitset->test(i)) {
                list->set(i);
            }
        }
        vec_index->SetBlacklist(list);

        auto& vector_param = context.query_ptr_->vectors.at(vector_placeholder);
        if (!vector_param->query_vector.float_data.empty()) {
            vector_param->nq = vector_param->query_vector.float_data.size() / vec_index->Dim();
        } else if (!vector_param->query_vector.binary_data.empty()) {
            vector_param->nq = vector_param->query_vector.binary_data.size() * 8 / vec_index->Dim();
        }

        status = VecSearch(context, context.query_ptr_->vectors.at(vector_placeholder), vec_index);
        if (!status.ok()) {
            return status;
        }
    } catch (std::exception& exception) {
        return Status{DB_ERROR, "Illegal search params"};
    }

    return Status::OK();
}

Status
ExecutionEngineImpl::SearchWithOptimizer(ExecutionEngineContext& context) {
    try {
        faiss::ConcurrentBitsetPtr bitset;
        std::string vector_placeholder;
        faiss::ConcurrentBitsetPtr list;
        faiss::ConcurrentBitsetPtr list_flat;

        SegmentPtr segment_ptr;
        segment_reader_->GetSegment(segment_ptr);
        knowhere::VecIndexPtr vec_index = nullptr;
        knowhere::VecIndexPtr vec_index_flat = nullptr;
        std::unordered_map<std::string, engine::DataType> attr_type;

        auto segment_visitor = segment_reader_->GetSegmentVisitor();
        auto field_visitors = segment_visitor->GetFieldVisitors();
        for (const auto& name : context.query_ptr_->index_fields) {
            auto field_visitor = segment_visitor->GetFieldVisitor(name);
            if (!field_visitor) {
                return Status(SERVER_INVALID_DSL_PARAMETER, "Field: " + name + " is not existed");
            }
            auto field = field_visitor->GetField();
            if (field->GetFtype() == static_cast<snapshot::FTYPE_TYPE>(engine::DataType::VECTOR_FLOAT) ||
                field->GetFtype() == static_cast<snapshot::FTYPE_TYPE>(engine::DataType::VECTOR_BINARY)) {
                STATUS_CHECK(segment_ptr->GetVectorIndex(name, vec_index));
                STATUS_CHECK(segment_ptr->GetVectorIndex(name + "_flat", vec_index_flat));
            } else {
                attr_type.insert(std::make_pair(name, static_cast<engine::DataType>(field->GetFtype())));
            }
        }

        list = vec_index->GetBlacklist();
        list_flat = vec_index_flat->GetBlacklist();
        entity_count_ = list->capacity();

        // Estimate score for optimizer to decide which strategy to use
        // Score is the percentage remain by scalar fields
        float score = 1.0f;
        auto status = Status::OK();

        auto strategy = context.query_ptr_->strategy;  // the strategy specified by DSL
        auto delta = context.query_ptr_->delta; // default to 2, the constant to multiply topK
        switch (strategy) {
            case 0: {
                STATUS_CHECK(EstimateScore(context.query_ptr_->root, attr_type, vector_placeholder, score));
                if (score <= 0.2) {
                    // strategy 3
                    status = StrategyThree(context, bitset, attr_type, vector_placeholder, list_flat, vec_index, delta);
                } else if (entity_count_ * (1 - score) <= 4096) {
                    // strategy 1
                    status = StrategyOne(context, bitset, attr_type, vector_placeholder, list_flat, vec_index_flat);
                } else {
                    // strategy 2
                    status = StrategyTwo(context, bitset, attr_type, vector_placeholder, list, vec_index);
                }
                break;
            }
            case 1: {
                status = StrategyOne(context, bitset, attr_type, vector_placeholder, list_flat, vec_index_flat);
                break;
            }
            case 2: {
                status = StrategyTwo(context, bitset, attr_type, vector_placeholder, list, vec_index);
                break;
            }
            case 3: {
                status = StrategyThree(context, bitset, attr_type, vector_placeholder, list_flat, vec_index, delta);
                break;
            }
            default: { break; }
        }
        if (!status.ok()) {
            return status;
        }
    } catch (std::exception& exception) {
        return Status{DB_ERROR, "Illegal search params"};
    }

    return Status::OK();
}

Status
ExecutionEngineImpl::EstimateScore(const query::GeneralQueryPtr& general_query,
                                   std::unordered_map<std::string, DataType>& attr_type,
                                   std::string& vector_placeholder, float& score) {
    Status status = Status::OK();
    if (general_query->leaf == nullptr) {
        float left_score = -1.0f;
        float right_score = -1.0f;
        if (general_query->bin->left_query != nullptr) {
            status = EstimateScore(general_query->bin->left_query, attr_type, vector_placeholder, left_score);
            if (!status.ok()) {
                return status;
            }
        }
        if (general_query->bin->right_query != nullptr) {
            status = EstimateScore(general_query->bin->right_query, attr_type, vector_placeholder, right_score);
            if (!status.ok()) {
                return status;
            }
        }

        if (left_score == -1 && right_score == -1) {
            score = 1.0f;
        } else if (left_score == -1 || right_score == -1) {
            score = left_score != -1 ? left_score : right_score;
        } else {
            switch (general_query->bin->relation) {
                case milvus::query::QueryRelation::AND:
                case milvus::query::QueryRelation::R1: {
                    score = left_score * right_score;
                    break;
                }
                case milvus::query::QueryRelation::OR:
                case milvus::query::QueryRelation::R2:
                case milvus::query::QueryRelation::R3: {
                    score = left_score + right_score;
                    break;
                }
                case milvus::query::QueryRelation::R4: {
                    score = left_score * (1 - right_score);
                    break;
                }
                default: {
                    std::string msg = "Invalid QueryRelation in BinaryQuery";
                    return Status{SERVER_INVALID_ARGUMENT, msg};
                }
            }
        }
        return status;
    } else {
        if (general_query->leaf->term_query != nullptr) {
            // bitset = std::make_shared<faiss::ConcurrentBitset>(entity_count_);
            // STATUS_CHECK(ProcessTermQuery(bitset, general_query->leaf->term_query, attr_type));
        }
        if (general_query->leaf->range_query != nullptr) {
            STATUS_CHECK(RangeQueryScore(general_query->leaf->range_query, attr_type, score));
        }
        if (!general_query->leaf->vector_placeholder.empty()) {
            // skip vector query
            // bitset = std::make_shared<faiss::ConcurrentBitset>(entity_count_, 255);
            vector_placeholder = general_query->leaf->vector_placeholder;
        }
    }
    return status;
}

Status
ExecutionEngineImpl::TermQueryScore(const milvus::query::TermQueryPtr& term_query,
                                    const std::unordered_map<std::string, DataType>& attr_type, float* score) {
    //    try {
    //        auto term_query_json = term_query->json_obj;
    //        JSON_NULL_CHECK(term_query_json);
    //        auto term_it = term_query_json.begin();
    //        if (term_it != term_query_json.end()) {
    //            const std::string& field_name = term_it.key();
    //            SegmentPtr segment_ptr;
    //            segment_reader_->GetSegment(segment_ptr);
    //            if (term_it.value().is_object()) {
    //                milvus::json term_values_json = term_it.value()["values"];
    //            } else {
    //            }
    //        }
    //        term_it++;
    //        if (term_it != term_query_json.end()) {
    //            return Status(SERVER_INVALID_DSL_PARAMETER, "Term query does not support multiple fields");
    //        }
    //    } catch (std::exception& ex) {
    //        return Status{SERVER_INVALID_DSL_PARAMETER, ex.what()};
    //    }
    //    return Status::OK();
}

template <typename T>
Status
ExecutionEngineImpl::ComputeRangeScore(const knowhere::IndexPtr& index_ptr, const milvus::engine::DataType& data_type,
                                       milvus::json& range_values_json, float& score) {
    try {
        auto T_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<T>>(index_ptr);

        T max = T_index->Max();
        T min = T_index->Min();
        bool has_gt = false, has_lt = false;
        T gt, lt;
        for (auto& range_value_it : range_values_json.items()) {
            const std::string& comp_op = range_value_it.key();
            T value = range_value_it.value();
            if (comp_op == "LT" || comp_op == "LE") {
                has_lt = true;
                lt = value;
            } else {
                has_gt = true;
                gt = value;
            }
        }
        if (not has_gt || gt < min) {
            gt = min;
        }
        if (not has_lt || lt > max) {
            lt = max;
        }
        score = (float)(lt - gt) / (max - min);
    } catch (std::exception& exception) {
        return Status{SERVER_INVALID_DSL_PARAMETER, exception.what()};
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::RangeQueryScore(const milvus::query::RangeQueryPtr& range_query,
                                     const std::unordered_map<std::string, DataType>& attr_type, float& score) {
    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);
    try {
        auto range_query_json = range_query->json_obj;
        JSON_NULL_CHECK(range_query_json);
        auto range_it = range_query_json.begin();
        if (range_it != range_query_json.end()) {
            const std::string& field_name = range_it.key();
            knowhere::IndexPtr index_ptr = nullptr;
            segment_ptr->GetStructuredIndex(field_name, index_ptr);
            auto type = attr_type.at(field_name);
            switch (type) {
                case DataType::INT32: {
                    STATUS_CHECK(ComputeRangeScore<int32_t>(index_ptr, type, range_it.value(), score));
                    break;
                }
                case DataType::INT64: {
                    STATUS_CHECK(ComputeRangeScore<int64_t>(index_ptr, type, range_it.value(), score));
                    break;
                }
                case DataType::FLOAT: {
                    STATUS_CHECK(ComputeRangeScore<float>(index_ptr, type, range_it.value(), score));
                    break;
                }
                case DataType::DOUBLE: {
                    STATUS_CHECK(ComputeRangeScore<double>(index_ptr, type, range_it.value(), score));
                    break;
                }
                default: { return Status(SERVER_INVALID_DSL_PARAMETER, "Range query field type is wrong"); }
            }
        }
        range_it++;
        if (range_it != range_query_json.end()) {
            return Status(SERVER_INVALID_DSL_PARAMETER, "Range query does not support multiple fields");
        }
    } catch (std::exception& ex) {
        return Status{SERVER_INVALID_DSL_PARAMETER, ex.what()};
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::StrategyOne(ExecutionEngineContext& context, faiss::ConcurrentBitsetPtr& bitset,
                                 std::unordered_map<std::string, engine::DataType>& attr_type,
                                 std::string& vector_placeholder, faiss::ConcurrentBitsetPtr& list,
                                 knowhere::VecIndexPtr& vec_index_flat) {
    auto status = ExecBinaryQuery(context.query_ptr_->root, bitset, attr_type, vector_placeholder);
    // Do And
    // make offset list
    std::vector<int64_t> offset;
    for (int64_t i = 0; i < entity_count_; i++) {
        if (!list->test(i) && bitset->test(i)) {
            offset.emplace_back(i);
        }
    }

    vec_index_flat->SetBlacklist(nullptr);

    auto& vector_param = context.query_ptr_->vectors.at(vector_placeholder);
    if (!vector_param->query_vector.float_data.empty()) {
        vector_param->nq = vector_param->query_vector.float_data.size() / vec_index_flat->Dim();
    } else if (!vector_param->query_vector.binary_data.empty()) {
        vector_param->nq = vector_param->query_vector.binary_data.size() * 8 / vec_index_flat->Dim();
    }

    status = VecSearchWithFlat(context, vector_param, vec_index_flat, offset);
    vec_index_flat->SetBlacklist(list);
    return status;
}

Status
ExecutionEngineImpl::StrategyTwo(ExecutionEngineContext& context, faiss::ConcurrentBitsetPtr& bitset,
                                 std::unordered_map<std::string, engine::DataType>& attr_type,
                                 std::string& vector_placeholder, faiss::ConcurrentBitsetPtr& list,
                                 knowhere::VecIndexPtr& vec_index) {
    auto status = ExecBinaryQuery(context.query_ptr_->root, bitset, attr_type, vector_placeholder);
    double t0 = getmillisecs();
    // Do And
    for (int64_t i = 0; i < entity_count_; i++) {
        if (!list->test(i) && !bitset->test(i)) {
            list->set(i);
        }
    }
    vec_index->SetBlacklist(list);
    printf("StrategyTwo SetBlacklist: %.2f\n", getmillisecs() - t0);

    auto& vector_param = context.query_ptr_->vectors.at(vector_placeholder);
    if (!vector_param->query_vector.float_data.empty()) {
        vector_param->nq = vector_param->query_vector.float_data.size() / vec_index->Dim();
    } else if (!vector_param->query_vector.binary_data.empty()) {
        vector_param->nq = vector_param->query_vector.binary_data.size() * 8 / vec_index->Dim();
    }

    t0 = getmillisecs();
    status = VecSearchWithOptimizer(context, vector_param, vec_index, false);
    printf("StrategyTwo VecSearchWithOptimizer: %.2f\n", getmillisecs() - t0);
    return status;
}

Status
ExecutionEngineImpl::StrategyThree(ExecutionEngineContext& context, faiss::ConcurrentBitsetPtr& bitset,
                                   std::unordered_map<std::string, engine::DataType>& attr_type,
                                   std::string& vector_placeholder, faiss::ConcurrentBitsetPtr& list,
                                   knowhere::VecIndexPtr& vec_index, float delta) {
    auto& vector_param = context.query_ptr_->vectors.begin()->second;
    if (!vector_param->query_vector.float_data.empty()) {
        vector_param->nq = vector_param->query_vector.float_data.size() / vec_index->Dim();
    } else if (!vector_param->query_vector.binary_data.empty()) {
        vector_param->nq = vector_param->query_vector.binary_data.size() * 8 / vec_index->Dim();
    }

    // printf("delta: %f\n", delta);
    vec_index->SetBlacklist(nullptr);
    double t0 = getmillisecs();
    auto status = VecSearchWithOptimizer(context, vector_param, vec_index, delta, true);
    printf("StrategyThree VecSearchWithOptimizer: %.2f\n", getmillisecs() - t0);
    if (!status.ok()) {
        return status;
    }
    vec_index->SetBlacklist(list);

    status = ExecBinaryQuery(context.query_ptr_->root, bitset, attr_type, vector_placeholder);

    auto uids = vec_index->GetUids();
    auto nq = vector_param->nq;
    auto topk = vector_param->topk;
    auto topk2 = (int64_t) (1.0 * topk * delta);

    auto& result_ids = context.query_result_->result_ids_;
    auto& result_distances = context.query_result_->result_distances_;

    t0 = getmillisecs();
    for (int i = 0; i < nq; i++) {
        int curr = i * topk;
        int invalid_count = 0;
        for (int j = 0; j < topk2; j++) {
            auto id = result_ids[i * topk2 + j];
            if (list->test(id) || !bitset->test(id)) {
                invalid_count++;
            } else {
                result_ids[curr] = id;
                result_distances[curr] = result_distances[i * topk2 + j];
                curr++;
            }
        }
        if (invalid_count + topk > topk2) {
            context.query_result_->result_ids_.clear();
            context.query_result_->result_distances_.clear();
            status = StrategyTwo(context, bitset, attr_type, vector_placeholder, list, vec_index);
            return status;
        }
    }

    int64_t num = nq * topk;
    result_ids.resize(num);
    result_distances.resize(num);

    /* map offsets to ids */
    for (int64_t i = 0; i < num; ++i) {
        int64_t offset = result_ids[i];
        if (offset != -1) {
            result_ids[i] = uids[offset];
        } else {
            result_ids[i] = -1;
        }
    }

    /* 
    std::vector<engine::ResultIds> ids(nq);
    std::vector<engine::ResultDistances> distances(nq);
    for (int i = 0; i < vector_param->nq; i++) {
        ids[i].resize(topk2);
        distances[i].resize(topk2);
        memcpy(ids[i].data(), result_ids.data() + i * topk2, topk2 * sizeof(faiss::Index::idx_t));
        memcpy(distances[i].data(), result_distances.data() + i * topk2, topk2 * sizeof(faiss::Index::distance_t));
    }

    for (int64_t i = result_ids.size() - 1; i >= 0; i--) {
        auto id = uid2off_.at(result_ids[i]);
        if (list->test(id) || !bitset->test(id)) {
            result_ids.erase(result_ids.begin() + i, result_ids.begin() + i + 1);
            result_distances.erase(result_distances.begin() + i, result_distances.begin() + i + 1);
            auto& cur_result_ids = ids[i / topk2];
            cur_result_ids.erase(cur_result_ids.begin() + i % topk2, cur_result_ids.begin() + i % topk2 + 1);
            auto& cur_result_dis = distances[i / topk2];
            cur_result_dis.erase(cur_result_dis.begin() + i % topk2, cur_result_dis.begin() + i % topk2 + 1);
        }
    }

    for (int64_t i = 0; i < nq; i++) {
        if (ids[i].size() < topk) {
            context.query_result_->result_ids_.clear();
            context.query_result_->result_distances_.clear();
            status = StrategyTwo(context, bitset, attr_type, vector_placeholder, list, vec_index);
            return status;
        } else if (ids[i].size() > topk) {
            auto remove_size = ids[i].size() - topk;
            result_ids.erase(result_ids.begin() + (i + 1) * topk, result_ids.begin() + (i + 1) * topk + remove_size);
            result_distances.erase(result_distances.begin() + (i + 1) * topk,
                                   result_distances.begin() + (i + 1) * topk + remove_size);
        }
    }
    */

    printf("StrategyThree filtering time: %.2f\n", getmillisecs() - t0);

    return Status::OK();
}

Status
ExecutionEngineImpl::ExecBinaryQuery(const milvus::query::GeneralQueryPtr& general_query,
                                     faiss::ConcurrentBitsetPtr& bitset,
                                     std::unordered_map<std::string, DataType>& attr_type,
                                     std::string& vector_placeholder) {
    Status status = Status::OK();
    if (general_query->leaf == nullptr) {
        faiss::ConcurrentBitsetPtr left_bitset, right_bitset;
        if (general_query->bin->left_query != nullptr) {
            status = ExecBinaryQuery(general_query->bin->left_query, left_bitset, attr_type, vector_placeholder);
            if (!status.ok()) {
                return status;
            }
        }
        if (general_query->bin->right_query != nullptr) {
            status = ExecBinaryQuery(general_query->bin->right_query, right_bitset, attr_type, vector_placeholder);
            if (!status.ok()) {
                return status;
            }
        }

        if (left_bitset == nullptr || right_bitset == nullptr) {
            bitset = left_bitset != nullptr ? left_bitset : right_bitset;
        } else {
            switch (general_query->bin->relation) {
                case milvus::query::QueryRelation::AND:
                case milvus::query::QueryRelation::R1: {
                    bitset = (*left_bitset) & right_bitset;
                    break;
                }
                case milvus::query::QueryRelation::OR:
                case milvus::query::QueryRelation::R2:
                case milvus::query::QueryRelation::R3: {
                    bitset = (*left_bitset) | right_bitset;
                    break;
                }
                case milvus::query::QueryRelation::R4: {
                    for (uint64_t i = 0; i < entity_count_; ++i) {
                        if (left_bitset->test(i) && !right_bitset->test(i)) {
                            bitset->set(i);
                        }
                    }
                    break;
                }
                default: {
                    std::string msg = "Invalid QueryRelation in BinaryQuery";
                    return Status{SERVER_INVALID_ARGUMENT, msg};
                }
            }
        }
        return status;
    } else {
        if (general_query->leaf->term_query != nullptr) {
            bitset = std::make_shared<faiss::ConcurrentBitset>(entity_count_);
            STATUS_CHECK(ProcessTermQuery(bitset, general_query->leaf->term_query, attr_type));
        }
        if (general_query->leaf->range_query != nullptr) {
            bitset = std::make_shared<faiss::ConcurrentBitset>(entity_count_);
            STATUS_CHECK(ProcessRangeQuery(attr_type, bitset, general_query->leaf->range_query));
        }
        if (!general_query->leaf->vector_placeholder.empty()) {
            // skip vector query
            bitset = std::make_shared<faiss::ConcurrentBitset>(entity_count_, 255);
            vector_placeholder = general_query->leaf->vector_placeholder;
        }
    }
    return status;
}

template <typename T>
Status
ProcessIndexedTermQuery(faiss::ConcurrentBitsetPtr& bitset, knowhere::IndexPtr& index_ptr,
                        milvus::json& term_values_json) {
    try {
        auto T_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<T>>(index_ptr);
        if (not T_index) {
            return Status{SERVER_INVALID_ARGUMENT, "Attribute's type is wrong"};
        }
        size_t term_size = term_values_json.size();
        std::vector<T> term_value(term_size);
        size_t offset = 0;
        for (auto& data : term_values_json) {
            term_value[offset] = data.get<T>();
            ++offset;
        }

        bitset = T_index->In(term_size, term_value.data());
    } catch (std::exception& exception) {
        return Status{SERVER_INVALID_DSL_PARAMETER, exception.what()};
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::IndexedTermQuery(faiss::ConcurrentBitsetPtr& bitset, const std::string& field_name,
                                      const DataType& data_type, milvus::json& term_values_json) {
    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);
    knowhere::IndexPtr index_ptr = nullptr;
    auto attr_index = segment_ptr->GetStructuredIndex(field_name, index_ptr);
    if (!index_ptr) {
        return Status(DB_ERROR, "Get field: " + field_name + " structured index failed");
    }
    switch (data_type) {
        case DataType::INT8: {
            STATUS_CHECK(ProcessIndexedTermQuery<int8_t>(bitset, index_ptr, term_values_json));
            break;
        }
        case DataType::INT16: {
            STATUS_CHECK(ProcessIndexedTermQuery<int16_t>(bitset, index_ptr, term_values_json));
            break;
        }
        case DataType::INT32: {
            STATUS_CHECK(ProcessIndexedTermQuery<int32_t>(bitset, index_ptr, term_values_json));
            break;
        }
        case DataType::INT64: {
            STATUS_CHECK(ProcessIndexedTermQuery<int64_t>(bitset, index_ptr, term_values_json));
            break;
        }
        case DataType::FLOAT: {
            STATUS_CHECK(ProcessIndexedTermQuery<float>(bitset, index_ptr, term_values_json));
            break;
        }
        case DataType::DOUBLE: {
            STATUS_CHECK(ProcessIndexedTermQuery<double>(bitset, index_ptr, term_values_json));
            break;
        }
        default: { return Status(SERVER_INVALID_ARGUMENT, "Attribute:" + field_name + " type is wrong"); }
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::ProcessTermQuery(faiss::ConcurrentBitsetPtr& bitset, const query::TermQueryPtr& term_query,
                                      std::unordered_map<std::string, DataType>& attr_type) {
    try {
        auto term_query_json = term_query->json_obj;
        JSON_NULL_CHECK(term_query_json);
        auto term_it = term_query_json.begin();
        if (term_it != term_query_json.end()) {
            const std::string& field_name = term_it.key();
            if (term_it.value().is_object()) {
                milvus::json term_values_json = term_it.value()["values"];
                STATUS_CHECK(IndexedTermQuery(bitset, field_name, attr_type.at(field_name), term_values_json));
            } else {
                STATUS_CHECK(IndexedTermQuery(bitset, field_name, attr_type.at(field_name), term_it.value()));
            }
        }
        term_it++;
        if (term_it != term_query_json.end()) {
            return Status(SERVER_INVALID_DSL_PARAMETER, "Term query does not support multiple fields");
        }
    } catch (std::exception& ex) {
        return Status{SERVER_INVALID_DSL_PARAMETER, ex.what()};
    }
    return Status::OK();
}

template <typename T>
Status
ProcessIndexedRangeQuery(faiss::ConcurrentBitsetPtr& bitset, knowhere::IndexPtr& index_ptr,
                         milvus::json& range_values_json) {
    try {
        auto T_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<T>>(index_ptr);

        bool flag = false;
        for (auto& range_value_it : range_values_json.items()) {
            const std::string& comp_op = range_value_it.key();
            T value = range_value_it.value();
            if (not flag) {
                bitset = (*bitset) | T_index->Range(value, knowhere::s_map_operator_type.at(comp_op));
                flag = true;
            } else {
                bitset = (*bitset) & T_index->Range(value, knowhere::s_map_operator_type.at(comp_op));
            }
        }
    } catch (std::exception& exception) {
        return Status{SERVER_INVALID_DSL_PARAMETER, exception.what()};
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::IndexedRangeQuery(faiss::ConcurrentBitsetPtr& bitset, const DataType& data_type,
                                       knowhere::IndexPtr& index_ptr, milvus::json& range_values_json) {
    auto status = Status::OK();
    switch (data_type) {
        case DataType::INT8: {
            STATUS_CHECK(ProcessIndexedRangeQuery<int8_t>(bitset, index_ptr, range_values_json));
            break;
        }
        case DataType::INT16: {
            STATUS_CHECK(ProcessIndexedRangeQuery<int16_t>(bitset, index_ptr, range_values_json));
            break;
        }
        case DataType::INT32: {
            STATUS_CHECK(ProcessIndexedRangeQuery<int32_t>(bitset, index_ptr, range_values_json));
            break;
        }
        case DataType::INT64: {
            STATUS_CHECK(ProcessIndexedRangeQuery<int64_t>(bitset, index_ptr, range_values_json));
            break;
        }
        case DataType::FLOAT: {
            STATUS_CHECK(ProcessIndexedRangeQuery<float>(bitset, index_ptr, range_values_json));
            break;
        }
        case DataType::DOUBLE: {
            STATUS_CHECK(ProcessIndexedRangeQuery<double>(bitset, index_ptr, range_values_json));
            break;
        }
        default:
            break;
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::ProcessRangeQuery(const std::unordered_map<std::string, DataType>& attr_type,
                                       faiss::ConcurrentBitsetPtr& bitset, const query::RangeQueryPtr& range_query) {
    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);
    try {
        auto range_query_json = range_query->json_obj;
        JSON_NULL_CHECK(range_query_json);
        auto range_it = range_query_json.begin();
        if (range_it != range_query_json.end()) {
            const std::string& field_name = range_it.key();
            knowhere::IndexPtr index_ptr = nullptr;
            segment_ptr->GetStructuredIndex(field_name, index_ptr);
            STATUS_CHECK(IndexedRangeQuery(bitset, attr_type.at(field_name), index_ptr, range_it.value()));
        }
        range_it++;
        if (range_it != range_query_json.end()) {
            return Status(SERVER_INVALID_DSL_PARAMETER, "Range query does not support multiple fields");
        }
    } catch (std::exception& ex) {
        return Status{SERVER_INVALID_DSL_PARAMETER, ex.what()};
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::BuildIndex() {
    TimeRecorderAuto rc("ExecutionEngineImpl::BuildIndex");

    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);

    auto segment_visitor = segment_reader_->GetSegmentVisitor();
    auto& snapshot = segment_visitor->GetSnapshot();
    auto& segment = segment_visitor->GetSegment();

    snapshot::OperationContext context;
    context.prev_partition = snapshot->GetResource<snapshot::Partition>(segment->GetPartitionId());
    auto build_op = std::make_shared<snapshot::ChangeSegmentFileOperation>(context, snapshot);

    for (auto& field_name : target_fields_) {
        // create snapshot segment files
        CollectionIndex index_info;
        auto status = CreateSnapshotIndexFile(build_op, field_name, index_info);
        if (!status.ok()) {
            return status;
        }

        // build index by knowhere
        auto field_visitor = segment_visitor->GetFieldVisitor(field_name);
        auto& field = field_visitor->GetField();

        auto root_path = segment_reader_->GetRootPath();
        auto op_ctx = build_op->GetContext();
        auto new_visitor = SegmentVisitor::Build(snapshot, segment, op_ctx.new_segment_files);
        auto segment_writer_ptr = std::make_shared<segment::SegmentWriter>(root_path, new_visitor);
        if (IsVectorField(field)) {
            knowhere::VecIndexPtr new_index;
            status = BuildKnowhereIndex(field_name, index_info, new_index);
            if (!status.ok()) {
                return status;
            }
            segment_writer_ptr->SetVectorIndex(field_name, new_index);

            rc.RecordSection("build vector index for field: " + field_name);

            // serialze index files
            status = segment_writer_ptr->WriteVectorIndex(field_name);
            if (!status.ok()) {
                return status;
            }

            rc.RecordSection("serialize vector index for field: " + field_name);
        } else {
            knowhere::IndexPtr index_ptr;
            segment_ptr->GetStructuredIndex(field_name, index_ptr);
            segment_writer_ptr->SetStructuredIndex(field_name, index_ptr);

            rc.RecordSection("build structured index for field: " + field_name);

            // serialze index files
            status = segment_writer_ptr->WriteStructuredIndex(field_name);
            if (!status.ok()) {
                return status;
            }

            rc.RecordSection("serialize structured index for field: " + field_name);
        }
    }

    // finish transaction
    build_op->Push();

    return Status::OK();
}

Status
ExecutionEngineImpl::GetSFParams(milvus::knowhere::IndexPtr& index_ptr, const milvus::engine::DataType& data_type,
                                 milvus::json& sf_params) {
    switch (data_type) {
        case DataType::INT32: {
            auto int32_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<int32_t>>(index_ptr);
            sf_params["max"] = int32_index->Max();
            sf_params["min"] = int32_index->Min();
            break;
        }
        case DataType::INT64: {
            auto int32_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<int32_t>>(index_ptr);
            sf_params["max"] = int32_index->Max();
            sf_params["min"] = int32_index->Min();
            break;
        }
        case DataType::FLOAT: {
            auto int32_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<int32_t>>(index_ptr);
            sf_params["max"] = int32_index->Max();
            sf_params["min"] = int32_index->Min();
            break;
        }
        case DataType::DOUBLE: {
            auto int32_index = std::dynamic_pointer_cast<knowhere::StructuredIndexSort<int32_t>>(index_ptr);
            sf_params["max"] = int32_index->Max();
            sf_params["min"] = int32_index->Min();
            break;
        }
        default: { return Status(SERVER_UNEXPECTED_ERROR, "Wrong data type"); }
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::CreateSnapshotIndexFile(AddSegmentFileOperation& operation, const std::string& field_name,
                                             CollectionIndex& index_info) {
    auto segment_visitor = segment_reader_->GetSegmentVisitor();
    auto& segment = segment_visitor->GetSegment();
    auto field_visitor = segment_visitor->GetFieldVisitor(field_name);
    auto& field = field_visitor->GetField();
    bool is_vector = IsVectorField(field);

    auto element_visitor = field_visitor->GetElementVisitor(engine::FieldElementType::FET_INDEX);
    if (element_visitor == nullptr) {
        return Status(DB_ERROR, "Could not build index: index not specified");  // no index specified
    }

    auto& index_element = element_visitor->GetElement();
    index_info.index_name_ = index_element->GetName();
    index_info.index_type_ = index_element->GetTypeName();
    auto params = index_element->GetParams();
    if (params.find(engine::PARAM_INDEX_METRIC_TYPE) != params.end()) {
        index_info.metric_name_ = params[engine::PARAM_INDEX_METRIC_TYPE];
    }
    if (params.find(engine::PARAM_INDEX_EXTRA_PARAMS) != params.end()) {
        index_info.extra_params_ = params[engine::PARAM_INDEX_EXTRA_PARAMS];
    }

    snapshot::SegmentFilePtr seg_file = element_visitor->GetFile();
    if (seg_file != nullptr) {
        // index already build?
        std::string file_path = engine::snapshot::GetResPath<engine::snapshot::SegmentFile>(
            segment_reader_->GetCollectionsPath(), seg_file);
        file_path +=
            (is_vector ? codec::VectorIndexFormat::FilePostfix() : codec::StructuredIndexFormat::FilePostfix());
        if (CommonUtil::IsFileExist(file_path)) {
            return Status(DB_ERROR, "Could not build index: Index file already exist");  // index already build
        }
    } else {
        // create snapshot index file
        snapshot::SegmentFileContext sf_context;
        sf_context.field_name = field_name;
        sf_context.field_element_name = index_element->GetName();
        sf_context.collection_id = segment->GetCollectionId();
        sf_context.partition_id = segment->GetPartitionId();
        sf_context.segment_id = segment->GetID();

        SegmentPtr segment_ptr;
        segment_reader_->GetSegment(segment_ptr);
        knowhere::IndexPtr index_ptr;
        segment_ptr->GetStructuredIndex(field_name, index_ptr);
        json sf_param;
        if (!IsVectorField(field) && index_ptr) {
            STATUS_CHECK(GetSFParams(index_ptr, field->GetFtype(), sf_param));
        }
        sf_context.params = sf_param;

        auto status = operation->CommitNewSegmentFile(sf_context, seg_file);
        if (!status.ok()) {
            return status;
        }
    }

    // create snapshot compress file
    if (utils::RequireCompressFile(index_info.index_type_)) {
        auto compress_visitor = field_visitor->GetElementVisitor(engine::FieldElementType::FET_COMPRESS);
        if (compress_visitor == nullptr) {
            return Status(DB_ERROR,
                          "Could not build index: compress element not exist");  // something wrong in CreateIndex
        }

        auto& compress_element = compress_visitor->GetElement();
        seg_file = compress_visitor->GetFile();
        if (seg_file == nullptr) {
            // create snapshot index compress file
            snapshot::SegmentFileContext sf_context;
            sf_context.field_name = field_name;
            sf_context.field_element_name = compress_element->GetName();
            sf_context.collection_id = segment->GetCollectionId();
            sf_context.partition_id = segment->GetPartitionId();
            sf_context.segment_id = segment->GetID();

            auto status = operation->CommitNewSegmentFile(sf_context, seg_file);
            if (!status.ok()) {
                return status;
            }
        }
    }

    return Status::OK();
}

Status
ExecutionEngineImpl::BuildKnowhereIndex(const std::string& field_name, const CollectionIndex& index_info,
                                        knowhere::VecIndexPtr& new_index) {
    SegmentPtr segment_ptr;
    segment_reader_->GetSegment(segment_ptr);

    knowhere::VecIndexPtr index_raw;
    segment_ptr->GetVectorIndex(field_name, index_raw);

    auto from_index = std::dynamic_pointer_cast<knowhere::IDMAP>(index_raw);
    auto bin_from_index = std::dynamic_pointer_cast<knowhere::BinaryIDMAP>(index_raw);
    if (from_index == nullptr && bin_from_index == nullptr) {
        LOG_ENGINE_ERROR_ << "ExecutionEngineImpl: from_index is not IDMAP, skip build index for " << field_name;
        throw Exception(DB_ERROR, "ExecutionEngineImpl: from_index is not IDMAP");
    }

    // build index by knowhere
    new_index = CreateVecIndex(index_info.index_type_);
    if (!new_index) {
        throw Exception(DB_ERROR, "Unsupported index type");
    }

    auto segment_visitor = segment_reader_->GetSegmentVisitor();
    auto& snapshot = segment_visitor->GetSnapshot();
    auto& segment = segment_visitor->GetSegment();
    auto field_visitor = segment_visitor->GetFieldVisitor(field_name);
    auto& field = field_visitor->GetField();

    auto field_json = field->GetParams();
    auto dimension = field_json[milvus::knowhere::meta::DIM];

    snapshot::SIZE_TYPE row_count;
    snapshot->GetSegmentRowCount(segment->GetID(), row_count);

    milvus::json conf = index_info.extra_params_;
    conf[knowhere::meta::DIM] = dimension;
    conf[knowhere::meta::ROWS] = row_count;
    conf[knowhere::meta::DEVICEID] = gpu_num_;
    conf[knowhere::Metric::TYPE] = index_info.metric_name_;
    LOG_ENGINE_DEBUG_ << "Index params: " << conf.dump();
    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(new_index->index_type());
    if (!adapter->CheckTrain(conf, new_index->index_mode())) {
        throw Exception(DB_ERROR, "Illegal index params");
    }
    LOG_ENGINE_DEBUG_ << "Index config: " << conf.dump();

    std::vector<idx_t> uids;
    faiss::ConcurrentBitsetPtr blacklist;
    knowhere::DatasetPtr dataset;
    if (from_index) {
        dataset =
            knowhere::GenDatasetWithIds(row_count, dimension, from_index->GetRawVectors(), from_index->GetRawIds());
        uids = from_index->GetUids();
        blacklist = from_index->GetBlacklist();
    } else if (bin_from_index) {
        dataset = knowhere::GenDatasetWithIds(row_count, dimension, bin_from_index->GetRawVectors(),
                                              bin_from_index->GetRawIds());
        uids = bin_from_index->GetUids();
        blacklist = bin_from_index->GetBlacklist();
    }

    try {
        new_index->BuildAll(dataset, conf);
    } catch (std::exception& ex) {
        std::string msg = "Knowhere failed to build index: " + std::string(ex.what());
        return Status(DB_ERROR, msg);
    }

#ifdef MILVUS_GPU_VERSION
    /* for GPU index, need copy back to CPU */
    if (new_index->index_mode() == knowhere::IndexMode::MODE_GPU) {
        auto device_index = std::dynamic_pointer_cast<knowhere::GPUIndex>(new_index);
        new_index = device_index->CopyGpuToCpu(conf);
    }
#endif

    new_index->SetUids(uids);
    if (blacklist != nullptr) {
        new_index->SetBlacklist(blacklist);
    }

    return Status::OK();
}

}  // namespace engine
}  // namespace milvus

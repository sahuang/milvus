// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <faiss/utils/ConcurrentBitset.h>
#include <memory>
#include <utility>
#include <vector>

#include "knowhere/common/Dataset.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Typedef.h"
#include "knowhere/index/Index.h"
#include "knowhere/index/IndexType.h"

namespace milvus {
namespace knowhere {

#define INDEX_DATA "INDEX_DATA"
#define RAW_DATA "RAW_DATA"
#define SQ8_DATA "SQ8_DATA"

class VecIndex : public Index {
 public:
    virtual std::unique_ptr<std::vector<int64_t>>
    BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
        Train(dataset_ptr, config);
        return Add(dataset_ptr, config);
    }

    virtual void
    Train(const DatasetPtr& dataset, const Config& config) = 0;

    virtual std::unique_ptr<std::vector<int64_t>>
    Add(const DatasetPtr& dataset, const Config& config) = 0;

    virtual void
    AddWithoutIds(const DatasetPtr& dataset, const Config& config) = 0;

    virtual DatasetPtr
    Query(const DatasetPtr& dataset, const Config& config) = 0;

#if 0
    virtual DatasetPtr
    QueryById(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }
#endif

    // virtual DatasetPtr
    // QueryByRange(const DatasetPtr&, const Config&) = 0;
    //
    // virtual MetricType
    // metric_type() = 0;

    virtual int64_t
    Dim() = 0;

    virtual int64_t
    Count() = 0;

    virtual IndexType
    index_type() const {
        return index_type_;
    }

    virtual IndexMode
    index_mode() const {
        return index_mode_;
    }

#if 0
    virtual DatasetPtr
    GetVectorById(const DatasetPtr& dataset, const Config& config) {
        return nullptr;
    }
#endif

    faiss::ConcurrentBitsetPtr
    GetBlacklist() {
        return bitset_;
    }

    void
    SetBlacklist(faiss::ConcurrentBitsetPtr bitset_ptr) {
        bitset_ = std::move(bitset_ptr);
    }

    const std::vector<IDType>&
    GetUids() const {
        return uids_;
    }

    void
    SetUids(std::vector<IDType>& uids) {
        uids_.clear();
        uids_.swap(uids);
    }

    size_t
    BlacklistSize() {
        if (bitset_) {
            return bitset_->size() * sizeof(uint8_t);
        } else {
            return 0;
        }
    }

    size_t
    UidsSize() {
        return uids_.size() * sizeof(IDType);
    }

    virtual int64_t
    IndexSize() {
        if (index_size_ == -1) {
            KNOWHERE_THROW_MSG("Index size not set");
        }
        return index_size_;
    }

    void
    SetIndexSize(int64_t size) {
        index_size_ = size;
    }

    virtual void
    UpdateIndexSize() {
    }

    int64_t
    Size() override {
        return BlacklistSize() + UidsSize() + IndexSize();
    }

 protected:
    IndexType index_type_ = "";
    IndexMode index_mode_ = IndexMode::MODE_CPU;
    faiss::ConcurrentBitsetPtr bitset_ = nullptr;
    std::vector<IDType> uids_;
    int64_t index_size_ = -1;
};

using VecIndexPtr = std::shared_ptr<VecIndex>;

}  // namespace knowhere
}  // namespace milvus

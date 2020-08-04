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

#pragma once

#include <SPTAG/AnnService/inc/Core/VectorIndex.h>

#include <cstdint>
#include <memory>
#include <string>

#include "knowhere/index/vector_index/VecIndex.h"

namespace milvus {
namespace knowhere {

class CPUSPTAGRNG : public VecIndex {
 public:
    explicit CPUSPTAGRNG(const std::string& IndexType);

 public:
    BinarySet
    Serialize(const Config& config = Config()) override;

    void
    Load(const BinarySet& index_array) override;

    std::unique_ptr<std::vector<int64_t>>
    BuildAll(const DatasetPtr& dataset_ptr, const Config& config) override {
        Train(dataset_ptr, config);
        return nullptr;
    }

    void
    Train(const DatasetPtr& dataset_ptr, const Config& config) override;

    std::unique_ptr<std::vector<int64_t>>
    Add(const DatasetPtr&, const Config&) override {
        KNOWHERE_THROW_MSG("Incremental index is not supported");
    }

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override {
        KNOWHERE_THROW_MSG("Incremental index is not supported");
    }

    DatasetPtr
    Query(const DatasetPtr& dataset_ptr, const Config& config) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    void
    UpdateIndexSize() override;

 private:
    void
    SetParameters(const Config& config);

 private:
    std::shared_ptr<SPTAG::VectorIndex> index_ptr_;
};

using CPUSPTAGRNGPtr = std::shared_ptr<CPUSPTAGRNG>;

}  // namespace knowhere
}  // namespace milvus

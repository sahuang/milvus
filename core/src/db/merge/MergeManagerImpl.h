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

#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "db/merge/MergeManager.h"
#include "db/merge/MergeStrategy.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {

class MergeManagerImpl : public MergeManager {
 public:
    explicit MergeManagerImpl(const DBOptions& options);

    Status
    MergeSegments(int64_t collection_id, MergeStrategyType type) override;

 private:
    Status
    CreateStrategy(MergeStrategyType type, MergeStrategyPtr& strategy);

 private:
    DBOptions options_;
};  // MergeManagerImpl

}  // namespace engine
}  // namespace milvus

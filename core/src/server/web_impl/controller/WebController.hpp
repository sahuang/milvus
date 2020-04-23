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

#include <string>
#include <iostream>

#include <oatpp/web/server/api/ApiController.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/core/macro/component.hpp>

#include "utils/Log.h"
#include "utils/TimeRecorder.h"

#include "server/web_impl/Constants.h"
#include "server/web_impl/dto/ConfigDto.hpp"
#include "server/web_impl/dto/IndexDto.hpp"
#include "server/web_impl/dto/PartitionDto.hpp"
#include "server/web_impl/dto/TableDto.hpp"
#include "server/web_impl/dto/VectorDto.hpp"
#include "server/web_impl/handler/WebRequestHandler.h"

namespace milvus {
namespace server {
namespace web {

class WebController : public oatpp::web::server::api::ApiController {
 public:
    WebController(const std::shared_ptr<ObjectMapper>& objectMapper)
        : oatpp::web::server::api::ApiController(objectMapper) {}

 public:
    static std::shared_ptr<WebController> createShared(
        OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper)) {
        return std::make_shared<WebController>(objectMapper);
    }

    /**
     *  Begin ENDPOINTs generation ('ApiController' codegen)
     */
#include OATPP_CODEGEN_BEGIN(ApiController)

    ADD_CORS(root)

    ENDPOINT("GET", "/", root) {
        auto response = createResponse(Status::CODE_200, "Welcome to milvus");
        response->putHeader(Header::CONTENT_TYPE, "text/plain");
        return response;
    }

    ADD_CORS(State)

    ENDPOINT("GET", "/state", State) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/state\'");
        tr.ElapseFromBegin("Total cost ");
        return createDtoResponse(Status::CODE_200, StatusDto::createShared());
    }

    ADD_CORS(GetDevices)

    ENDPOINT("GET", "/devices", GetDevices) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/devices\'");
        tr.RecordSection("Receive request");

        auto devices_dto = DevicesDto::createShared();
        WebRequestHandler handler = WebRequestHandler();
        auto status_dto = handler.GetDevices(devices_dto);
        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, devices_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(AdvancedConfigOptions)

    ENDPOINT("OPTIONS", "/config/advanced", AdvancedConfigOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(GetAdvancedConfig)

    ENDPOINT("GET", "/config/advanced", GetAdvancedConfig) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/config/advanced\'");
        tr.RecordSection("Received request.");

        auto config_dto = AdvancedConfigDto::createShared();
        WebRequestHandler handler = WebRequestHandler();
        auto status_dto = handler.GetAdvancedConfig(config_dto);

        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, config_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(SetAdvancedConfig)

    ENDPOINT("PUT", "/config/advanced", SetAdvancedConfig, BODY_DTO(AdvancedConfigDto::ObjectWrapper, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "PUT \'/config/advanced\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.SetAdvancedConfig(body);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

#ifdef MILVUS_GPU_VERSION

    ADD_CORS(GPUConfigOptions)

    ENDPOINT("OPTIONS", "/config/gpu_resources", GPUConfigOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(GetGPUConfig)

    ENDPOINT("GET", "/config/gpu_resources", GetGPUConfig) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/config/gpu_resources\'");
        tr.RecordSection("Received request");

        auto gpu_config_dto = GPUConfigDto::createShared();
        WebRequestHandler handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.GetGpuConfig(gpu_config_dto);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, gpu_config_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(SetGPUConfig)

    ENDPOINT("PUT", "/config/gpu_resources", SetGPUConfig, BODY_DTO(GPUConfigDto::ObjectWrapper, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "PUT \'/config/gpu_resources\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();
        auto status_dto = handler.SetGpuConfig(body);

        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                   + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);
        return response;
    }

#endif

    ADD_CORS(TablesOptions)

    ENDPOINT("OPTIONS", "/collections", TablesOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(CreateTable)

    ENDPOINT("POST", "/collections", CreateTable, BODY_DTO(TableRequestDto::ObjectWrapper, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "POST \'/collections\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.CreateTable(body);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_201, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);
        return response;
    }

    ADD_CORS(ShowTables)

    ENDPOINT("GET", "/collections", ShowTables, QUERIES(const QueryParams&, query_params)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/collections\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();


        String result;
        auto status_dto = handler.ShowTables(query_params, result);
        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, result);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(TableOptions)

    ENDPOINT("OPTIONS", "/collections/{collection_name}", TableOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(GetTable)

    ENDPOINT("GET", "/collections/{collection_name}", GetTable,
        PATH(String, collection_name), QUERIES(const QueryParams&, query_params)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/collections/" + collection_name->std_str() + "\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();

        String response_str;
        auto status_dto = handler.GetTable(collection_name, query_params, response_str);

        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, response_str);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(DropTable)

    ENDPOINT("DELETE", "/collections/{collection_name}", DropTable, PATH(String, collection_name)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "DELETE \'/collections/" + collection_name->std_str() + "\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.DropTable(collection_name);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_204, status_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(IndexOptions)

    ENDPOINT("OPTIONS", "/collections/{collection_name}/indexes", IndexOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(CreateIndex)

    ENDPOINT("POST", "/collections/{collection_name}/indexes", CreateIndex,
             PATH(String, collection_name), BODY_STRING(String, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "POST \'/tables/" + collection_name->std_str() + "/indexes\'");
        tr.RecordSection("Received request.");

        auto handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.CreateIndex(collection_name, body);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_201, status_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(GetIndex)

    ENDPOINT("GET", "/collections/{collection_name}/indexes", GetIndex, PATH(String, collection_name)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/collections/" + collection_name->std_str() + "/indexes\'");
        tr.RecordSection("Received request.");

        auto handler = WebRequestHandler();

        OString result;
        auto status_dto = handler.GetIndex(collection_name, result);

        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, result);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(DropIndex)

    ENDPOINT("DELETE", "/collections/{collection_name}/indexes", DropIndex, PATH(String, collection_name)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "DELETE \'/collections/" + collection_name->std_str() + "/indexes\'");
        tr.RecordSection("Received request.");

        auto handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.DropIndex(collection_name);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_204, status_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        std::string ttr = "Done. Status: code = " + std::to_string(status_dto->code->getValue())
                          + ", reason = " + status_dto->message->std_str() + ". Total cost";
        tr.ElapseFromBegin(ttr);

        return response;
    }

    ADD_CORS(PartitionsOptions)

    ENDPOINT("OPTIONS", "/collections/{collection_name}/partitions", PartitionsOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(CreatePartition)

    ENDPOINT("POST", "/collections/{collection_name}/partitions",
             CreatePartition, PATH(String, collection_name), BODY_DTO(PartitionRequestDto::ObjectWrapper, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "POST \'/collections/" + collection_name->std_str() + "/partitions\'");
        tr.RecordSection("Received request.");

        auto handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.CreatePartition(collection_name, body);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_201, status_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(ShowPartitions)

    ENDPOINT("GET", "/collections/{collection_name}/partitions", ShowPartitions,
             PATH(String, collection_name), QUERIES(const QueryParams&, query_params)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/collections/" + collection_name->std_str() + "/partitions\'");
        tr.RecordSection("Received request.");

        auto offset = query_params.get("offset");
        auto page_size = query_params.get("page_size");

        auto partition_list_dto = PartitionListDto::createShared();
        auto handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.ShowPartitions(collection_name, query_params, partition_list_dto);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_200, partition_list_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(DropPartition)

    ENDPOINT("DELETE", "/collections/{collection_name}/partitions", DropPartition,
             PATH(String, collection_name), BODY_STRING(String, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) +
                        "DELETE \'/collections/" + collection_name->std_str() + "/partitions\'");
        tr.RecordSection("Received request.");

        auto handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.DropPartition(collection_name, body);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_204, status_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(ShowSegments)

    ENDPOINT("GET", "/collections/{collection_name}/segments", ShowSegments,
             PATH(String, collection_name), QUERIES(const QueryParams&, query_params)) {
        auto offset = query_params.get("offset");
        auto page_size = query_params.get("page_size");

        auto handler = WebRequestHandler();
        String response;
        auto status_dto = handler.ShowSegments(collection_name, query_params, response);

        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                return createResponse(Status::CODE_200, response);
            case StatusCode::COLLECTION_NOT_EXISTS:
                return createDtoResponse(Status::CODE_404, status_dto);
            default:
                return createDtoResponse(Status::CODE_400, status_dto);
        }
    }

    ADD_CORS(GetSegmentInfo)
    /**
     *
     * GetSegmentVector
     */
    ENDPOINT("GET", "/collections/{collection_name}/segments/{segment_name}/{info}", GetSegmentInfo,
             PATH(String, collection_name), PATH(String, segment_name), PATH(String, info), QUERIES(const QueryParams&, query_params)) {
        auto offset = query_params.get("offset");
        auto page_size = query_params.get("page_size");

        auto handler = WebRequestHandler();
        String response;
        auto status_dto = handler.GetSegmentInfo(collection_name, segment_name, info, query_params, response);

        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                return createResponse(Status::CODE_200, response);
            case StatusCode::COLLECTION_NOT_EXISTS:
                return createDtoResponse(Status::CODE_404, status_dto);
            default:
                return createDtoResponse(Status::CODE_400, status_dto);
        }
    }

    ADD_CORS(VectorsOptions)

    ENDPOINT("OPTIONS", "/collections/{collection_name}/vectors", VectorsOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(GetVectors)
    /**
     *
     * GetVectorByID ?id=
     */
    ENDPOINT("GET", "/collections/{collection_name}/vectors", GetVectors,
             PATH(String, collection_name), QUERIES(const QueryParams&, query_params)) {
        auto handler = WebRequestHandler();
        String response;
        auto status_dto = handler.GetVector(collection_name, query_params, response);

        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                return createResponse(Status::CODE_200, response);
            case StatusCode::COLLECTION_NOT_EXISTS:
                return createDtoResponse(Status::CODE_404, status_dto);
            default:
                return createDtoResponse(Status::CODE_400, status_dto);
        }
    }

    ADD_CORS(Insert)

    ENDPOINT("POST", "/collections/{collection_name}/vectors", Insert,
             PATH(String, collection_name), BODY_STRING(String, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "POST \'/collections/" + collection_name->std_str() + "/vectors\'");
        tr.RecordSection("Received request.");

        auto ids_dto = VectorIdsDto::createShared();
        WebRequestHandler handler = WebRequestHandler();

        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.Insert(collection_name, body, ids_dto);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createDtoResponse(Status::CODE_201, ids_dto);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(VectorsOp)

    ENDPOINT("PUT", "/collections/{collection_name}/vectors", VectorsOp,
             PATH(String, collection_name), BODY_STRING(String, body)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "PUT \'/collections/" + collection_name->std_str() + "/vectors\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();

        OString result;
        std::shared_ptr<OutgoingResponse> response;
        auto status_dto = handler.VectorsOp(collection_name, body, result);
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, result);
                break;
            case StatusCode::COLLECTION_NOT_EXISTS:
                response = createDtoResponse(Status::CODE_404, status_dto);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(SystemOptions)

    ENDPOINT("OPTIONS", "/system/{info}", SystemOptions) {
        return createResponse(Status::CODE_204, "No Content");
    }

    ADD_CORS(SystemInfo)

    ENDPOINT("GET", "/system/{info}", SystemInfo, PATH(String, info), QUERIES(const QueryParams&, query_params)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "GET \'/system/" + info->std_str() + "\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();
        OString result = "";
        auto status_dto = handler.SystemInfo(info, query_params, result);
        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, result);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }

        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

    ADD_CORS(SystemOp)

    ENDPOINT("PUT", "/system/{op}", SystemOp, PATH(String, op), BODY_STRING(String, body_str)) {
        TimeRecorder tr(std::string(WEB_LOG_PREFIX) + "PUT \'/system/" + op->std_str() + "\'");
        tr.RecordSection("Received request.");

        WebRequestHandler handler = WebRequestHandler();
        handler.RegisterRequestHandler(::milvus::server::RequestHandler());

        String response_str;
        auto status_dto = handler.SystemOp(op, body_str, response_str);

        std::shared_ptr<OutgoingResponse> response;
        switch (status_dto->code->getValue()) {
            case StatusCode::SUCCESS:
                response = createResponse(Status::CODE_200, response_str);
                break;
            default:
                response = createDtoResponse(Status::CODE_400, status_dto);
        }
        tr.ElapseFromBegin("Done. Status: code = " + std::to_string(status_dto->code->getValue())
                           + ", reason = " + status_dto->message->std_str() + ". Total cost");

        return response;
    }

/**
 *  Finish ENDPOINTs generation ('ApiController' codegen)
 */
#include OATPP_CODEGEN_END(ApiController)

};

} // namespace web
} // namespace server
} // namespace milvus

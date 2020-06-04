/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include <iostream>

#include "faiss/IndexIVF.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"
#include "faiss/gpu/GpuIndexFlat.h"
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuAutoTune.h"
#include "faiss/gpu/GpuCloner.h"
#include "faiss/gpu/GpuClonerOptions.h"
#include "faiss/gpu/GpuIndexIVF.h"

#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/AuxIndexStructures.h"

#include "faiss/IndexFlat.h"
#include "faiss/VectorTransform.h"
#include "faiss/IndexLSH.h"
#include "faiss/IndexPQ.h"

#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFSpectralHash.h"
#include "faiss/MetaIndexes.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/IndexHNSW.h"
#include "faiss/OnDiskInvertedLists.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryFromFloat.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/utils/distances.h"
#include "faiss/index_factory.h"

using namespace faiss;

#define PRINT_RESULT 0

void print_result(const char* unit, long number, long k, long nq, long *I) {
    printf("%s: I (2 first results)=\n", unit);
    for(int i = 0; i < number; i++) {
        for(int j = 0; j < k; j++)
            printf("%5ld ", I[i * k + j]);
        printf("\n");
    }

    printf("%s: I (2 last results)=\n", unit);
    for(int i = nq - number; i < nq; i++) {
        for(int j = 0; j < k; j++)
            printf("%5ld ", I[i * k + j]);
        printf("\n");
    }
}


int main() {
    const char* filename = "index500k.index";
    
#if PRINT_RESULT
    int number = 8;
#endif

    int d = 128;                            // dimension
    int nq = 10000;                        // nb of queries
    int nprobe = 20;
    float *xq = new float[d * nq];
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) {
            xq[d * i + j] = drand48();
        }
    }
    faiss::distance_compute_blas_threshold = 800;

    faiss::gpu::StandardGpuResources res;

    int k = 4;
    std::shared_ptr<faiss::Index> gpu_index_ivf_ptr;

    const char* index_description = "IVF4096,Flat";
//     const char* index_description = "IVF3276,SQ8";

    faiss::Index *cpu_index = nullptr;
    faiss::IndexIVF* cpu_ivf_index = nullptr;

    long nb = 50000;                       // database size
//        printf("-----------------------\n");
    long size = d * nb;
    float *xb = new float[size];
    memset(xb, 0, size * sizeof(float));
    printf("size: %ld\n", (size * sizeof(float)) );
    for(long i = 0; i < nb; i++) {
        for(long j = 0; j < d; j++) {
            float rand = drand48();
            xb[d * i + j] = rand;
        }
    }

    faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);
    auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

    gpu_index_ivf_ptr = std::shared_ptr<faiss::Index>(device_index);

    assert(!device_index->is_trained);
    device_index->train(nb, xb);
    assert(device_index->is_trained);
    device_index->add(nb, xb);  // add vectors to the index

    printf("is_trained = %s\n", device_index->is_trained ? "true" : "false");
    printf("ntotal = %ld\n", device_index->ntotal);

    cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));
    faiss::write_index(cpu_index, filename);
    printf("index.index is stored successfully.\n");

    cpu_ivf_index = dynamic_cast<faiss::IndexIVF*>(cpu_index);
    if(cpu_ivf_index != nullptr) {
        cpu_ivf_index->to_readonly();
    }

    auto init_gpu =[&](int device_id, faiss::gpu::GpuClonerOptions* option) {
        option->allInGpu = true;
        faiss::Index* tmp_index = faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index, option);
        delete tmp_index;
    };

    auto gpu_executor = [&](int device_id, float *xb, faiss::gpu::GpuClonerOptions* option) {
        auto tmp_index = faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index, option);
        delete tmp_index;
        double t0 = getmillisecs ();
        option->allInGpu = true;
        
        {
            // cpu to gpu
            option->allInGpu = true;

            tmp_index = faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index, option);
            gpu_index_ivf_ptr = std::shared_ptr<faiss::Index>(tmp_index);
        }
        
        double t1 = getmillisecs ();
        printf("CPU to GPU loading time: %0.2f\n", t1 - t0);

        {
            long *I = new long[k * nq];
            float *D = new float[k * nq];
        if(option->allInGpu) {
            faiss::gpu::GpuIndexIVF* gpu_index_ivf =
            dynamic_cast<faiss::gpu::GpuIndexIVF*>(gpu_index_ivf_ptr.get());
            gpu_index_ivf->setNumProbes(nprobe);
            for(long i = 0; i < 1; ++ i) {
                double t2 = getmillisecs();

                auto xxx = faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index, option);
                t2 = getmillisecs();
                auto cpuindex = faiss::gpu::index_gpu_to_cpu_without_codes (xxx);

                printf("gpu to cpu without codes time: %0.2f\n", getmillisecs() - t2);
                t2 = getmillisecs();

                auto cpuindex2 = faiss::gpu::index_gpu_to_cpu (xxx);

                printf("gpu to cpu time: %0.2f\n", getmillisecs() - t2);
                t2 = getmillisecs();

                faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(cpuindex);
                ivf_index->nprobe = nprobe;
                ivf_index->search_without_codes(nq, xq, xb, k, D, I);
                printf("cpu without codes execution time: %0.2f\n", getmillisecs() - t2);
                t2 = getmillisecs();

                // cpu to gpu
                auto cpu_to_gpu = faiss::gpu::index_cpu_to_gpu_without_codes(&res, device_id, cpu_index, xb, option);
                printf("cpu to gpu without codes time: %0.2f\n", getmillisecs() - t2);
                t2 = getmillisecs();
                auto cpu_to_gpu_ivf_ptr = std::shared_ptr<faiss::Index>(cpu_to_gpu);
                faiss::gpu::GpuIndexIVF* gpu_index_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(cpu_to_gpu_ivf_ptr.get());
                gpu_index_ivf->setNumProbes(nprobe);
                gpu_index_ivf->search_without_codes(nq, xq, xb, k, D, I);
                printf("gpu without codes execution time: %0.2f\n", getmillisecs() - t2);
                printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%.5f ", D[i * k + j]);
            printf("\n");
        }
                t2 = getmillisecs();
            }
        } else {
            faiss::IndexIVFScalarQuantizer* index_ivf =
            dynamic_cast<faiss::IndexIVFScalarQuantizer*>(gpu_index_ivf_ptr.get());
            index_ivf->nprobe = nprobe;
            for(long i = 0; i < 1; ++ i) {
                double t2 = getmillisecs();
                index_ivf->search(nq, xq, k, D, I);
                double t3 = getmillisecs();
                printf("- GPU: %d, execution time: %0.2f\n", device_id, t3 - t2);
            }
        }

            // print results
#if PRINT_RESULT
            print_result("GPU", number, k, nq, I);
#endif
            delete [] I;
            delete [] D;
        }
        double t4 = getmillisecs();

        printf("GPU:%d total time: %0.2f\n", device_id, t4 - t0);

    };
    printf("----------------------------------\n");
    auto cpu_executor = [&]() {       // search xq
        printf("CPU: \n");
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        double t4 = getmillisecs();
        faiss::IndexIVF* ivf_index =
            dynamic_cast<faiss::IndexIVF*>(cpu_index);
        ivf_index->nprobe = nprobe;
        cpu_index->search(nq, xq, k, D, I);
        double t5 = getmillisecs();
        printf("CPU execution time: %0.2f\n", t5 - t4);
        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%.5f ", D[i * k + j]);
            printf("\n");
        }
#if PRINT_RESULT
        print_result("CPU", number, k, nq, I);
#endif
        delete [] I;
        delete [] D;
    };

    for(long i = 0; i < 1; ++ i) {
        cpu_executor();
    }

    faiss::gpu::GpuClonerOptions option0;
    //faiss::gpu::GpuClonerOptions option1;

//    init_gpu(0, &option0);
//    init_gpu(1, &option1);

//    double tx = getmillisecs();
    std::thread t1(gpu_executor, 0, xb, &option0);
    //std::thread t2(gpu_executor, 1, &option1);
    t1.join();
    //t2.join();
//    double ty = getmillisecs();
//    printf("Total GPU execution time: %0.2f\n", ty - tx);

    delete [] xq;
    return 0;
}

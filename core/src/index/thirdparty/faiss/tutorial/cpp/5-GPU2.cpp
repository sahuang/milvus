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



int main() {

    int d = 128;                            // dimension
    int nq = 2048;                        // nb of queries
    int nprobe = 32;
    float *xq = new float[d * nq];
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) {
            xq[d * i + j] = drand48();
        }
    }

    faiss::distance_compute_blas_threshold = 800;
    faiss::gpu::StandardGpuResources res;

    int k = 4;

    const char* index_description = "IVF4096,Flat";

    faiss::Index *cpu_index = nullptr;
    faiss::IndexIVF* cpu_ivf_index = nullptr;

    long nb = 4000000;                       // database size
    long size = d * nb;
    float *xb = new float[size];
    memset(xb, 0, size * sizeof(float));
    for(long i = 0; i < nb; i++) {
        for(long j = 0; j < d; j++) {
            float rand = drand48();
            xb[d * i + j] = rand;
        }
    }

    faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);
    auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

    assert(!device_index->is_trained);
    device_index->train(nb, xb);
    assert(device_index->is_trained);
    device_index->add(nb, xb);  // add vectors to the index

    printf("is_trained = %s\n", device_index->is_trained ? "true" : "false");
    printf("ntotal = %ld\n", device_index->ntotal);

    cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));

    cpu_ivf_index = dynamic_cast<faiss::IndexIVF*>(cpu_index);
    if(cpu_ivf_index != nullptr) {
        cpu_ivf_index->to_readonly();
    }

    auto gpu_executor = [&](int device_id, float *xb, faiss::gpu::GpuClonerOptions* option) {

        option->allInGpu = true;
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        double t2 = getmillisecs();
        auto cpu_to_gpu = faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index, option);
        printf("cpu to gpu time: %0.2f\n", getmillisecs() - t2);
        t2 = getmillisecs();

        auto cpuindex = faiss::gpu::index_gpu_to_cpu(cpu_to_gpu);
        printf("gpu to cpu time: %0.2f\n", getmillisecs() - t2);

        faiss::IndexIVF* ivf_index = dynamic_cast<faiss::IndexIVF*>(cpuindex);
        ivf_index->nprobe = nprobe;
        t2 = getmillisecs();
        ivf_index->search(nq, xq, k, D, I);
        printf("cpu execution time: %0.2f\n", getmillisecs() - t2);

        auto cpu_to_gpu_ivf_ptr = std::shared_ptr<faiss::Index>(cpu_to_gpu);
        faiss::gpu::GpuIndexIVF* gpu_index_ivf = dynamic_cast<faiss::gpu::GpuIndexIVF*>(cpu_to_gpu_ivf_ptr.get());
        gpu_index_ivf->setNumProbes(nprobe);
        t2 = getmillisecs();
        gpu_index_ivf->search(nq, xq, k, D, I);
        printf("gpu execution time: %0.2f\n", getmillisecs() - t2);

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

        delete [] I;
        delete [] D;
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

        delete [] I;
        delete [] D;
    };

    for(long i = 0; i < 1; ++ i) {
        cpu_executor();
    }

    faiss::gpu::GpuClonerOptions option0;

    std::thread t1(gpu_executor, 0, xb, &option0);
    t1.join();

    delete [] xq;
    return 0;
}

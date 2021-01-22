/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexRHNSW.h>
#include <faiss/FaissHook.h>

float * base_read(const char*fname, size_t k, long nb)
{
    FILE *f = fopen(fname, "r");
    float *out  = new float[k*nb];
    fread(out, sizeof(float), k*nb, f);
    fclose(f);
    return out;
}

int main()
{
    size_t d = 128;
    size_t nb = 10000000;

    size_t loops = 1;

    float *xb = base_read("sift10M_base", d, nb);

    size_t nq = 10000;
    float *xq;

    xq = base_read("sift10M_query", d, nq);

    size_t k = 100;

    faiss::distance_compute_blas_threshold = 10000000;
    size_t nlist = 1024;

    printf("IVF_FLAT index search\n");
    faiss::IndexFlat coarse_quantizer(d);
    auto index = new faiss::IndexIVFFlat(&coarse_quantizer, d, nlist);
    index->train(nb, xb);
    index->add(nb, xb);

    // Obtain centroid data and calculate radius
    auto ails = dynamic_cast<faiss::ArrayInvertedLists*>(index->invlists);
    auto ids = ails->ids;
    auto centroids = dynamic_cast<faiss::IndexFlat*>(index->quantizer)->xb;
    printf("%.2f %.2f %.2f\n", centroids[0], centroids[10], centroids[1024]);

    std::vector<float> radius;
    for (size_t i = 0; i < nlist; i++) {
        float *center = centroids.data() + d * i * sizeof(float);
        auto ids_i = ids[i];
        auto res = 0.0f;
        for (size_t j = 0; j < ids_i.size(); j++) {
            printf("ids_i[j]: %ld\n", ids_i[j]);
            float *data = xb + d * ids_i[j] * sizeof(float);
            float dis = faiss::fvec_L2sqr (center, data, d);
            if (dis > res) res = dis;
        }
        radius.push_back(res);
        if (i % 50 == 0) printf("Currently at %ld with radius %.2f...\n", i, res);
    }
    printf("First 2 and last 2 radius: %.2f %.2f %.2f %.2f\n", radius[0], radius[1], radius[nlist-2], radius[nlist-1]);

    // Given a query, generate nlist distances and write to a file
    for (size_t i = 0; i < 100; i++) {
        auto query = xq + i * d * sizeof(float);
        std::ofstream MyFile;
        MyFile.open("/tmp/server_file.txt", std::ios_base::app);
        for (size_t j = 0; j < nlist; j++) {
            float D = faiss::fvec_L2sqr (query, centroids.data() + d * j * sizeof(float), d) - radius[j];
            MyFile << D << std::endl;
        }
        MyFile.close();
    }

    delete [] xq;
    return 0;
}
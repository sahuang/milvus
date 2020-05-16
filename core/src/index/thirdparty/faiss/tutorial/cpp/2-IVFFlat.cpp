/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <faiss/utils/utils.h>


int main() {
    int d = 128;                            // dimension
    int nb = 5000000;                       // database size
    int nq = 10000;                        // nb of queries

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    srand48(1234567);

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }


    int nlist = 4096;
    int k = 4;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
    // here we specify METRIC_L2, by default it performs inner-product search
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    double t0 = faiss::getmillisecs();
    index.add_without_codes(nb, xb);
    printf("Add time: %.2f\n", faiss::getmillisecs() - t0);
    t0 = faiss::getmillisecs();

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index.nprobe = 20;
        t0 = faiss::getmillisecs();
        index.search_without_codes(nq, xq, xb, k, D, I);
        printf("Search time: %.2f\n", faiss::getmillisecs() - t0);
        t0 = faiss::getmillisecs();

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
    }



    delete [] xb;
    delete [] xq;

    return 0;
}

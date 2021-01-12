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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_factory.h>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

float
CalcRecall(int64_t topk, int64_t k, int nq, faiss::Index::idx_t* gt, faiss::Index::idx_t* nt) {
    float sum_ratio = 0.0f;
    for (int i = 0; i < nq; i++) {
        //std::vector<int64_t> ids_0 = true_ids[i].ids;
        //std::vector<int64_t> ids_1 = result_ids[i].ids;
        std::vector<faiss::Index::idx_t> ids_0(gt+i*k,gt+i*k+topk);
        std::vector<faiss::Index::idx_t> ids_1(nt+i*topk,nt+i*topk+topk);
        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());
        std::vector<faiss::Index::idx_t> v(nq * 2);
        std::vector<faiss::Index::idx_t>::iterator it;
        it=std::set_intersection (ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it-v.begin());
        sum_ratio += 1.0f * v.size() / topk;
    }
    return 1.0 * sum_ratio / nq;
}


int main()
{
    size_t d;
    size_t nb;

    size_t loops = 3;

    float *xb = fvecs_read("sift1M/sift_base.fvecs", &d, &nb);

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb);

    size_t nq;
    float *xq;

    size_t d2;
    xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
    assert(d == d2 || !"query does not have same dimension as train set");

    size_t k; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    // load ground-truth and convert int to long
    size_t nq2;
    int *gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
    assert(nq2 == nq || !"incorrect nb of ground truth entries");

    gt = new faiss::Index::idx_t[k * nq];
    for(int i = 0; i < k * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete [] gt_int;

    faiss::distance_compute_blas_threshold = 10000000;

    printf("OPQ16_64,IVF4096,PQ16 index search\n");
    for (size_t nprobe = 1; nprobe <= 4096; nprobe *= 2) {
        {
            auto index = faiss::index_factory(d, "OPQ16_64,IVF4096,PQ16");
            long *I = new long[k * nq];
            float *D = new float[k * nq];
            auto ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(index);
            ivfpq->nprobe = nprobe;
            ivfpq->pq.M = 16;
            index->train(nb, xb);
            index->add(nb, xb);
            index->search(nq, xq, k, D, I);
            double avg = 0.0f;
            for (int i = 0; i < loops; i++) {
                double t0 = elapsed();
                index->search(nq, xq, k, D, I);
                avg += elapsed() - t0;
            }
            avg /= loops;
            printf("nprobe: %ld, OPQ Recall 1/10/100: %.4f/%.4f/%.4f, time spent: %.3fs\n", 
                nprobe, 
                CalcRecall(1, k, nq, gt, I), 
                CalcRecall(10, k, nq, gt, I), 
                CalcRecall(100, k, nq, gt, I), 
                avg);
            delete [] I;
            delete [] D;
        }
    }
    delete [] xq;
    delete [] gt;
    return 0;
}

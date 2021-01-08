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
#include <faiss/IndexHNSW.h>

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

float * base_read(const char*fname, size_t k, long nb)
{
    FILE *f = fopen(fname, "r");
    float *out  = new float[k*nb];
    fread(out, sizeof(float), k*nb, f);
    fclose(f);
    return out;
}

int * ground_read(const char*fname, size_t k, long nb)
{
    FILE *f = fopen(fname, "r");
    int *out  = new int[k*nb];
    fread(out, sizeof(int), k*nb, f);
    fclose(f);
    return out;
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
    size_t d = 128;
    size_t nb = 10000000;

    size_t loops = 1;

    float *xb = base_read("sift10M_base", d, nb);

    size_t nq = 10000;
    float *xq;

    xq = base_read("sift10M_query", d, nq);
    // assert(d == d2 || !"query does not have same dimension as train set");

    size_t k = 100; // nb of results per query in the GT

    faiss::distance_compute_blas_threshold = 10000000;
    size_t small_k = 100;
    // size_t nprobe = 32;
    size_t M = 32;
    size_t nlist = 65536;

    printf("FLAT search\n");
    faiss::IndexFlatL2 index(d);           // call constructor
    index.add(nb, xb);
    long *gt = new long[k * nq];
    float *D = new float[k * nq];
    index.search(nq, xq, k, D, gt);
    delete [] D;
/*
    printf("IVF_FLAT index search\n");
    for (size_t nprobe = 128; nprobe < 200; nprobe += 64) {
        {
            long *I = new long[small_k * nq];
            float *D = new float[small_k * nq];
            faiss::IndexFlatL2 quantizer(d);
            auto ivf = new faiss::IndexIVFFlat(&quantizer, d, nlist);
            auto ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(ivf);
            ivf_index->nprobe = nprobe;
            ivf->train(nb, xb);
            ivf->add(nb, xb);
	    ivf->search(nq, xq, small_k, D, I);
            double avg = 0.0f;
	    for (int i = 0; i < loops; i++) {
	        double t0 = elapsed();
	        ivf->search(nq, xq, small_k, D, I);
                avg += elapsed() - t0;
            }
            avg /= loops;
            printf("nprobe: %ld, IVF_FLAT Recall: %.4f, time spent: %.3fs\n", nprobe, CalcRecall(small_k, k, nq, gt, I), avg);
            delete [] I;
            delete [] D;
        }
    }
*/

    printf("IVF65536_HNSW32 index search\n");
    for (size_t nprobe = 256; nprobe < 512; nprobe += 16) {
        if (nprobe != 256 && nprobe != 328) continue;
        {
            faiss::IndexHNSWFlat coarse_quantizer(d, M, faiss::METRIC_L2);
            auto index = new faiss::IndexIVFFlat(&coarse_quantizer, d, nlist);
            long *I = new long[small_k * nq];
            float *D = new float[small_k * nq];
            index->nprobe = nprobe;
            index->train(nb, xb);
            index->add(nb, xb);
            index->search(nq, xq, small_k, D, I);
            double avg = 0.0f;
            for (int i = 0; i < loops; i++) {
                double t0 = elapsed();
                index->search(nq, xq, small_k, D, I);
                avg += elapsed() - t0;
            }
            avg /= loops;
            printf("nprobe: %ld, IVF_HNSW Recall: %.4f, time spent: %.3fs\n", nprobe, CalcRecall(small_k, k, nq, gt, I), avg);
            delete [] I;
            delete [] D;
        }
    }
    delete [] xq;
    delete [] gt;
    return 0;
}

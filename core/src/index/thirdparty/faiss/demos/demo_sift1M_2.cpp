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
#include <faiss/IndexScalarQuantizer.h>

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
        std::vector<faiss::Index::idx_t> ids_1(nt+i*k,nt+i*k+topk);
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
    faiss::IndexFlatL2 index(d);           // call constructor
    // printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);                     // add vectors to the index
    // printf("ntotal = %ld\n", index.ntotal);

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

    // FLAT index search
    /*
    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        index.search(nq, xq, k, D, I);
        double avg = 0.0f;
        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            index.search(nq, xq, k, D, I);
            avg += elapsed() - t0;
        }
        avg /= loops;
        printf("Flat Recall: %.4f, time spent: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg);
        delete [] I;
        delete [] D;
    }
    */

    // First IVF_SQ8, then IndexRefineFlat
    size_t nlist = 1024;
    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        faiss::IndexFlatL2 quantizer(d);
        faiss::Index* sq = new faiss::IndexIVFScalarQuantizer(&quantizer, d, nlist, faiss::QuantizerType::QT_8bit);
        auto ivfsq_index = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(sq);
        ivfsq_index->nprobe = 20;
        faiss::IndexRefineFlat rf = faiss::IndexRefineFlat(sq);
        rf.train(nb, xb);
        rf.add(nb, xb);
        rf.k_factor = 4;
        rf.search(nq, xq, k, D, I);
        double avg = 0.0f;
        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            rf.search(nq, xq, k, D, I);
            avg += elapsed() - t0;
        }
        avg /= loops;
        printf("RefineFlat Recall: %.4f, time spent: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg);
        delete [] I;
        delete [] D;
    }

    // IVF_FLAT alone
    {
        long *I = new long[k * nq];
        float *D = new float[k * nq];
        faiss::IndexFlatL2 quantizer(d);
        auto sq = new faiss::IndexIVFFlat(&quantizer, d, nlist);
        auto ivfsq_index = dynamic_cast<faiss::IndexIVFFlat*>(sq);
        ivfsq_index->nprobe = nlist;
        sq->train(nb, xb);
        sq->add(nb, xb);
        sq->search(nq, xq, k, D, I);
        double avg = 0.0f;
        for (int i = 0; i < loops; i++) {
            double t0 = elapsed();
            sq->search(nq, xq, k, D, I);
            avg += elapsed() - t0;
        }
        avg /= loops;
        printf("IVF_FLAT Recall: %.4f, time spent: %.3fs\n", CalcRecall(k, k, nq, gt, I), avg);
        delete [] I;
        delete [] D;
    }

    delete [] xq;
    delete [] gt;
    return 0;
}

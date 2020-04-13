/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/FaissHook.h>
#include <faiss/utils/random.h>

namespace faiss {

/**************************************************
 * Random data generation functions
 **************************************************/

RandomGenerator::RandomGenerator (int64_t seed)
    : mt((unsigned int)seed) {}

int RandomGenerator::rand_int ()
{
    return mt() & 0x7fffffff;
}

int64_t RandomGenerator::rand_int64 ()
{
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
}

int RandomGenerator::rand_int (int max)
{
    return mt() % max;
}

float RandomGenerator::rand_float ()
{
    return mt() / float(mt.max());
}

double RandomGenerator::rand_double ()
{
    return mt() / double(mt.max());
}


/***********************************************************************
 * Random functions in this C file only exist because Torch
 *  counterparts are slow and not multi-threaded.  Typical use is for
 *  more than 1-100 billion values. */


/* Generate a set of random floating point values such that x[i] in [0,1]
   multi-threading. For this reason, we rely on re-entreant functions.  */
void float_rand (float * x, size_t n, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++)
            x[i] = rng.rand_float ();
    }
}


void float_randn (float * x, size_t n, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {
        RandomGenerator rng (a0 + j * b0);

        double a = 0, b = 0, s = 0;
        int state = 0;  /* generate two number per "do-while" loop */

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        for (size_t i = istart; i < iend; i++) {
            /* Marsaglia's method (see Knuth) */
            if (state == 0) {
                do {
                    a = 2.0 * rng.rand_double () - 1;
                    b = 2.0 * rng.rand_double () - 1;
                    s = a * a + b * b;
                } while (s >= 1.0);
                x[i] = a * sqrt(-2.0 * log(s) / s);
            }
            else
                x[i] = b * sqrt(-2.0 * log(s) / s);
            state = 1 - state;
        }
    }
}


/* Integer versions */
void int64_rand (int64_t * x, size_t n, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;
        for (size_t i = istart; i < iend; i++)
            x[i] = rng.rand_int64 ();
    }
}

void int64_rand_max (int64_t * x, size_t n, uint64_t max, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;
        for (size_t i = istart; i < iend; i++)
            x[i] = rng.rand_int64 () % max;
    }
}


void rand_perm (int *perm, size_t n, int64_t seed)
{
    for (size_t i = 0; i < n; i++) perm[i] = i;

    RandomGenerator rng (seed);

    for (size_t i = 0; i + 1 < n; i++) {
        int i2 = i + rng.rand_int (n - i);
        std::swap(perm[i], perm[i2]);
    }
}


float nearest_center (int * perm, const float * x, size_t currId, size_t d, size_t numCentroids) {
    float minDist = std::numeric_limits<float>::max();

	for (int k = 0; k < numCentroids; k++) {
		float dis = fvec_L2sqr (x + currId * d, x + perm[k] * d, d);
		if (dis < minDist) {
			minDist = dis;
		}
	}

	return minDist;
}


void rand_perm_plus_plus (int * perm, const float * x, size_t k, size_t n, size_t d, int64_t seed) 
{
    RandomGenerator rng (seed);
    std::vector<float> nearestDis(n, 0);
    std::vector<float> totalProb(n, 0);
    size_t numCentroids = 0;

    // first step: init first centroid
    perm[0] = rng.rand_int (n);
    ++numCentroids;

    // select other k-1 centroids
    for (size_t i = 1; i < k; i++) {
        float sum = 0.0;
#pragma omp parallel for reduction (+ : sum)
        for (size_t p = 0; p < n; p++) {
            float minDistance = nearest_center(perm, x, p, d, numCentroids);
            sum += minDistance;
            nearestDis[p] = minDistance;
        }

        // Roulette Wheel Selection
        float rand_num = rng.rand_float () * 1000;

        // Obtain totalProb from D^2
        totalProb[0] = nearestDis[0] * 1000 / sum;
        if (totalProb[0] >= rand_num) {
            perm[i] = 0;
        } else {
            for (size_t p = 1; p < n; p++) {
                totalProb[p] = nearestDis[p] * 1000 / sum + totalProb[p - 1];
                if (totalProb[p] >= rand_num) {
                    perm[i] = p;
                    break;
                }
            }
        }

        ++numCentroids;
    }
}


void byte_rand (uint8_t * x, size_t n, int64_t seed)
{
    // only try to parallelize on large enough arrays
    const size_t nblock = n < 1024 ? 1 : 1024;

    RandomGenerator rng0 (seed);
    int a0 = rng0.rand_int (), b0 = rng0.rand_int ();

#pragma omp parallel for
    for (size_t j = 0; j < nblock; j++) {

        RandomGenerator rng (a0 + j * b0);

        const size_t istart = j * n / nblock;
        const size_t iend = (j + 1) * n / nblock;

        size_t i;
        for (i = istart; i < iend; i++)
            x[i] = rng.rand_int64 ();
    }
}

} // namespace faiss

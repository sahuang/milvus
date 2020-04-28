#include <faiss/FaissHook.h>

#include <faiss/utils/kmeans.h>
#include <faiss/utils/random.h>
#include <faiss/utils/distances.h>

namespace faiss {

    size_t chain_length = 20;

void get_candidates (size_t n, std::vector<float> q, std::vector<size_t>& cand_ind, RandomGenerator rng) {

    std::vector<float> v(chain_length);
    for (size_t i = 0; i < chain_length; i++) {
        v[i] = rng.rand_float ();
    }

    for (size_t i = 0; i < chain_length; i++) {
        float target = v[i];
        float curr = 0.0;

        for (size_t p = 0; p < n; p++) {
            curr += q[p];
            if (curr >= target) {
                cand_ind[i] = p;
                break;
            }
        }
    }
}

void kmeans_mc2_l2 (int * perm, const float * x, size_t k, size_t n, size_t d, int64_t seed, bool afk_mc2) {
    RandomGenerator rng0 (seed);
    size_t a0 = rng0.rand_int (), b0 = rng0.rand_int ();
    size_t centroids = 0;

    // Sample first center and compute proposal
    perm[0] = rng0.rand_int (n);
    ++centroids;

    std::vector<float> q(n, 1.0 / n);
    std::vector<float> di(n, 0);
    float sum_di = 0.0;
    float sum_q = 0.0;

    if (afk_mc2) {
#pragma omp parallel for reduction (+ : sum_di)
        for (size_t p = 0; p < n; p++) {
            di[p] = fvec_L2sqr (x + p * d, x + perm[0] * d, d);
            sum_di += di[p];
        }

#pragma omp parallel for reduction (+ : sum_q)
        for (size_t p = 0; p < n; p++) {
            q[p] = di[p] / sum_di + 1.0 / n;
            sum_q += q[p];
        }

#pragma omp parallel for
        for (size_t p = 0; p < n; p++) {
            // Renormalize the proposal distribution
            q[p] /= sum_q;
        }

    }

    // select other k-1 centroids
    for (size_t i = 1; i < k; i++) {
        RandomGenerator rng (a0 + i * b0);

        // choose chain_length candidates
        std::vector<size_t> cand_ind(chain_length, 0);
        std::vector<float> q_cand(chain_length, 0.0);
        get_candidates(n, q, cand_ind, rng);

        for (size_t p = 0; p < chain_length; p++) {
            q_cand[p] = q[cand_ind[p]];
        }

        // Compute pairwise distances
        std::vector<float> p_cand(chain_length, 0.0);
        for (size_t p = 0; p < chain_length; p++) {
            float min_dist = 1.0 / 0.0;
#pragma omp parallel for reduction (min : min_dist)
            for (size_t c = 0; c < centroids; c++) {
                float dis = fvec_L2sqr (x + cand_ind[p] * d, x + perm[c] * d, d);
                if (dis < min_dist) {
                    min_dist = dis;
                }
            }
            p_cand[p] = min_dist;
        }

        // Compute acceptance probabilities
        std::vector<float> rand_a(chain_length, 0.0);
        for (size_t p = 0; p < chain_length; p++) {
            rand_a[p] = rng.rand_float();
        }

        // Markov chain
        size_t curr_ind = 0;
        float curr_prob = p_cand[0] / q_cand[0];
        for (size_t j = 1; j < chain_length; j++) {
            float cand_prob = p_cand[j] / q_cand[j];
            if (curr_prob == 0.0 || cand_prob / curr_prob > rand_a[j]) {
                curr_ind = j;
                curr_prob = cand_prob;
            }
        }

        perm[centroids] = cand_ind[curr_ind];
        // printf("Centroid %d is %d\n", centroids, perm[centroids]);
        centroids++;
    }
}


} // namespace faiss
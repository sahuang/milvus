#pragma once

#include <stdint.h>
#include <faiss/utils/random.h>

namespace faiss {

extern size_t chain_length;

/** sample chain_length numbers from [0, n) with probability distribution q.
   *
   * @param n           number upper bound (exclusive)
   * @param q           prob distribution
   * @param cand_ind    output of size chain_length
   */
void get_candidates (size_t n, std::vector<float> q, std::vector<size_t>& cand_ind, faiss::RandomGenerator rng);

/** kmeans with mcmc
   *
   * @param perm        store results of k initial mc2 centroids
   * @param x           input data of size n * d
   * @param k           number of cluster centroids
   * @param seed        seed used for randomness
   * @param afk_mc2     whether to use afk mc^2 or vanilla mc^2
   */
void kmeans_mc2_l2 (int * perm, const float * x, size_t k, size_t n, size_t d, int64_t seed, bool afk_mc2);

} // namespace faiss
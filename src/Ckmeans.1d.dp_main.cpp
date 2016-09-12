/*
 *  Ckmeans.1d.dp_pymain.cpp --- wrapper function for "kmeans_1d_dp()"
 * 
 *  Greg Werbin
 *  Rocketrip
 *  July 2016
 * 
 * x:        An array containing input data to be clustered.
 * length:   Length of the one dimensional array.
 * minK:     Minimum number of clusters.
 * maxK:     Maximum number of clusters.
 * cluster:  An array of cluster IDs for each point in x.
 * centers:  An array of centers for each cluster.
 * withinss: An array of within-cluster sum of squares for each cluster.
 * size:     An array of sizes of each cluster.
 */

#include "Ckmeans.1d.dp.h"

extern "C" {
    /*
     * void Ckmeans_1d_dp(std::vector<double> x, int n,
     *                    std::vector<double> weights, int n_weights,
     *                    int min_k, int max_k,
     *                    std::vector<int> clustering, std::vector<double> centers,
     *                    std::vector<int> sizes, std::vector<double> withinss)
     * {
     *     if(*n_weights != *n) { y = 0; }
     *     kmeans_1d_dp(*x, (size_t)*n, weights,
     *                  (size_t)(*min_k), (size_t)(*max_k),
     *                  cluster, centers, within_ss, sizes);
     * }
     */
     void Ckmeans_1d_dp(double* x, int* length, double* y, int* ylength,
                       int* minK, int* maxK, int* cluster,
                       double* centers, double* withinss, int* size)
    {
        if(*ylength != *length) { y = 0; }
        kmeans_1d_dp(x, (size_t)*length, y, (size_t)(*minK), (size_t)(*maxK),
                     cluster, centers, withinss, size);
    }
}

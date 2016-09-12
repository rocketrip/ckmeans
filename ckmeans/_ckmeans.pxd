

cdef extern from "../src/Ckmeans.1d.dp_main.cpp":
    # void Ckmeans_1d_dp(vector[double] x, int n,
    #                    vector[double] weights, int n_weights,
    #                    int min_k, int max_k,
    #                    vector[int] clustering, vector[double] centers,
    #                    vector[int] sizes, vector[double] withinss)
    void Ckmeans_1d_dp(double *x, int* length, double *y, int * ylength,
                       int* minK, int *maxK, int* cluster,
                       double* centers, double* withinss, int* size)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char **argv) {

    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int dim[2], period[2], reorder;
    double sqrt_numprocs = sqrt(numprocs);
    if (( rank==0 ) && ( (int)sqrt_numprocs*sqrt_numprocs != numprocs )) {
        printf("[ERROR] Number of processors '%d' is not a perfect square.\n", numprocs);
        exit(1);
    }

    int n = atoi(argv[1]);
    int chunkarea = (n+1)*(n+1) / numprocs;
    int chunklength = sqrt(chunkarea);
    int localN = chunklength*chunklength;
    if (( rank==0 ) && ( ((n+1)*(n+1) % numprocs) != 0 )) {
        printf("[ERROR] Number of points '%d+1' not divisible by number of processors '%d'.\n", n, numprocs);
        exit(1);
    }

    // Set up cartesian topology of procs.
    MPI_Comm cartcomm;
    int cartsize = (int) sqrt_numprocs;
    dim[0] = dim[1] = cartsize;
    period[0] = period[1] = 0;
    reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartcomm);

    int tags[numprocs-1];
    for (int i=0; i<(numprocs-1); ++i) { tags[i] = i; }

    // Initialize stencil.
    int my_stencil[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
    stencil_struct* stencil = (stencil_struct*) malloc(sizeof(stencil));
    stencil->size = 3;
    stencil->extent = stencil->size/2;
    stencil->stencil = (int*) malloc(stencil->size*stencil->size*sizeof(int));
    memcpy(stencil->stencil, my_stencil, sizeof(my_stencil));

    // Initialize local arrays.
    d_struct* locald = init_locald(n, chunklength, rank, cartcomm, cartsize);
    //print_local2dmesh(chunklength, chunklength, locald->locald, rank, cartcomm);
    double* localg = init_localg(chunklength, locald->locald);
    print_local2dmesh(chunklength, chunklength, localg, rank, cartcomm);
    //double* localu = (double*) calloc(chunk*(n+1),sizeof(double));
    //double* localq = (double*) calloc(chunk*(n+1),sizeof(double));

    //int i;
    //double q0, tau, q1, beta;

    //// Step 2: q0 = dot(g,g)
    //MPI_Barrier(MPI_COMM_WORLD);
    //double runtime = MPI_Wtime();
    //dot(chunk, n+1, localg, localg, rank, MPI_COMM_WORLD, &q0);
    //for (int iter=0; iter<MAX_ITERS; iter++) {
    //    //printf("[RANK %d, Iter %d]\n", rank, iter);

    //    // Step 4: q = Ad. Exchange boundaries first
    //    // before applying the stencil.
    //    apply_stencil(n, stencil, locald, localq, rank, numprocs, chunk);

    //    // Step 5: tau = q0/dot(d,q)
    //    dot(chunk, n+1, locald->locald, localq, rank, MPI_COMM_WORLD, &tau);
    //    tau = q0/tau;

    //    // Step 6: u = u + tau*d
    //    for (i=0; i<localN; ++i) { localu[i] += tau*(locald->locald[i]); }

    //    // Step 7: g = g + tau*q
    //    for (i=0; i<localN; ++i) { localg[i] += tau*localq[i]; }

    //    // Step 8: q1 = dot(g,g)
    //    dot(chunk, n+1, localg, localg, rank, MPI_COMM_WORLD, &q1);
    //    beta = q1/q0;

    //    // Step 10: d = -g + beta*d
    //    for (i=0; i<localN; ++i) { locald->locald[i] = beta*(locald->locald[i]) - localg[i]; }
    //    q0 = q1;
    //}
    //runtime = MPI_Wtime() - runtime;
    //double max_runtime;
    //MPI_Reduce(&runtime, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //// Output: norm(g) = sqrt( dot(g,g) )
    //double norm_g;
    //dot(chunk, n+1, localg, localg, rank, MPI_COMM_WORLD, &norm_g);
    //norm_g = sqrt(norm_g);
    //if (rank==0) {
    //    printf("[INFO] norm_g = %.16lf\n", norm_g);
    //    printf("%.16lf\n", max_runtime);
    //}

    //free(localu); free(localg); free(localq);
    //free(locald);
    MPI_Finalize();

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

double* init_d(int n) {
    double h = (double) 1 / n;
    double *d = (double*) malloc((n+1)*(n+1)*sizeof(double));

    // Initialize d. No need to optimize this, since it's
    // only done once (at the start of the program).
    double xi, yj;
    for (int i=0; i<=n; ++i) {
        xi = i*h;
        for (int j=0; j<=n; ++j) {
            yj = j*h;
            // Account for boundary conditions.
            if (i==0 || i==n || j==0 || j==n) {
                d[i*(n+1)+j] = 0.0;
            }
            else {
                d[i*(n+1)+j] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }
        }
    }
    return d;
}

double* init_g(int n, double* d) {
    double* g = (double*) malloc((n+1)*(n+1)*sizeof(double));

    // Initialize g. No need to optimize this, since it's
    // only done once (at the start of the program).
    for (int i=0; i<=n; ++i) {
        for (int j=0; j<=n; ++j) {
            g[i*(n+1)+j] = -d[i*(n+1)+j];
        }
    }
    return g;
}

void print_2dmesh(int n, double* mesh) {
    for (int i=0; i<=n; ++i) {
        for (int j=0; j<=n; ++j) {
            printf("%lf ", mesh[i*(n+1)+j]);
        }
        putchar('\n');
    }
}

void apply_stencil(int n, stencil_struct my_stencil, double* src, double* dest) {
    int stencil_size = my_stencil.size;
    int extent = my_stencil.extent;
    int* stencil = my_stencil.stencil;

    double result;
    int i,j,l,m;
    int index;
    for (i=extent; i<(n+1)-extent; ++i) {
        for (j=extent; j<(n+1)-extent; ++j) {

            result = 0;
            for (l=0; l<stencil_size; ++l) {
                for (m=0; m<stencil_size; ++m) {
                    index = (i - extent + l)*(n+1) + (j - extent + m);
                    result += stencil[l*stencil_size+m] * src[index];
                }
            }
            dest[i*(n+1)+j] = result;
        }
    }
}

double dot(int N, double* v, double* w) {
    double sum = 0.0;
    for (int i=0; i<N; ++i) { sum += v[i]*w[i]; }
    return sum;
}

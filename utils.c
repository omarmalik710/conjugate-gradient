#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* init_d(int n) {
    double h = (double) 1 / n;
    double *d = (double*) malloc((n+1)*(n+1)*sizeof(double));

    // Initialize d. No need to optimize this, since it's
    // only done once (at the start of the program).
    double xi, yj;
    for (int j=0; j<=n; ++j) {
        yj = j*h;
        for (int i=0; i<=n; ++i) {
            xi = i*h;
            // Account for boundary conditions.
            if (i==0 || i==n || j==0 || j==n) {
                d[i+j*(n+1)] = 0.0;
            }
            else {
                d[i+j*(n+1)] = 2*h*h * ( xi*(1-xi) + yj*(1-yj) );
            }
        }
    }
    return d;
}

double* init_g(int n, double* d) {
    double* g = (double*) malloc((n+1)*(n+1)*sizeof(double));

    // Initialize g. No need to optimize this, since it's
    // only done once (at the start of the program).
    for (int j=0; j<=n; ++j) {
        for (int i=0; i<=n; ++i) {
            g[i+j*(n+1)] = -d[i+j*(n+1)];
        }
    }
    return g;
}

void print_2dmesh(int n, double* mesh) {
    for (int i=0; i<(n+1); ++i) {
        for (int j=0; j<(n+1); ++j) {
            printf("%lf ", mesh[i+j*(n+1)]);
        }
        putchar('\n');
    }
}

double* matvect_mult(int n, double* stencil, double* v) {
    double* w = (double*) calloc(n,sizeof(double));
    for (int j=0; j<n; ++j) {
        for (int i=0; i<n; ++i) {
            w[i] += stencil[i+j*n]*v[j];
        }
    }
    return w;
}

double dot(int n, double* v, double* w) {
    double sum = 0.0;
    for (int i=0; i<n; ++i) { sum += v[i]*w[i]; }
    return sum;
}

double* vect_add(int n, double* u, double* v) {
    double* w = (double*) malloc(n*sizeof(double));
    for (int i=0; i<n; ++i) { w[i] = u[i]+v[i]; }
    return w;
}

double* scal_mult(int n, double* v, double k) {
    double* w = (double*) malloc(n*sizeof(double));
    for (int i=0; i<n; ++i) { w[i] = k*v[i]; }
    return w;
}
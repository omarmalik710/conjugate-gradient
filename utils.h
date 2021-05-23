#define TOL 1e-5

typedef struct my_stencil {
    int size;
    int extent;
    int* stencil;
} stencil_struct;

double* init_d(int n);
double* init_g(int n, double* d);
void print_2dmesh(int n, double* mesh);
void apply_stencil(int n, stencil_struct my_stencil, double* src, double* dest);
double dot(int n, double* v, double* w);
double* vect_add(int n, double* u, double* v);
double* vect_sub(int n, double* u, double* v);
double* scal_mult(int n, double* v, double k);
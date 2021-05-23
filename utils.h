#define TOL 1e-5

double* init_d(int n);
double* init_g(int n, double* d);
void print_2dmesh(int n, double* mesh);
double* matvect_mult(int n, double* A, double* v);
double dot(int n, double* v, double* w);
double* vect_add(int n, double* u, double* v);
double* scal_mult(int n, double* v, double k);
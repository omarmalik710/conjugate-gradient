#define MAX_ITERS 1
typedef struct my_stencil {
    int size;
    int extent;
    int* stencil;
} stencil_struct;

typedef struct my_d {
    double* left_pad;
    double* top_pad;
    double* locald;
    double* bottom_pad;
    double* right_pad;
    double* corner_pad;
} d_struct;

typedef struct my_MPI_Settings {
    MPI_Comm cartcomm;
    MPI_Datatype rowtype;
    MPI_Datatype coltype;
    MPI_Request* irequests;
    MPI_Request* jrequests;
    MPI_Status status;
    int* itags;
    int* jtags;
    int cartsize;
} MPI_Settings;

//typedef struct my_corner_indices {
//    int icorn;
//    int jcorn;
//    int lstart;
//    int lend;
//    int mstart;
//    int mend;
//} corner_indices;

MPI_Settings* init_mpi_settings(int numprocs, int chunklength);
d_struct* init_locald(int n, int chunklength, int wrank, MPI_Settings* mpi_settings);
void exchange_boundaries(int chunklength, d_struct* locald, int myrank, MPI_Settings* mpi_settings);
double* init_localg(int chunklength, double* d);
void print_local2dmesh(int rows, int cols, double* mesh, int wrank, MPI_Comm cartcomm);
void apply_stencil(int chunklength, stencil_struct* my_stencil, d_struct* locald, double* localq, int myrank, MPI_Settings* mpi_settings);
void dot(int rows, int cols, double* localv, double* localw, MPI_Comm comm, double* result);
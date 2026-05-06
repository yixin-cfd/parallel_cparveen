#include <petsc.h>

int main(int argc, char **argv)
{
   PetscMPIInt    rank, size;
   PetscCall(PetscInitialize(&argc,&argv,NULL,"Hello World!"));
   PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
   PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, "Hello from rank %d of %d\n",rank,size));
   return PetscFinalize();
}

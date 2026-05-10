// Wasilij Barsukow, An Active Flux method for the Euler equations based on the
// exact acoustic evolution operator, https://arxiv.org/abs/2506.03291
#define BC_LEFT   periodic
#define BC_RIGHT  periodic
#define BC_BOTTOM periodic
#define BC_TOP    periodic

const double xmin =  0.0, xmax = 2.0;
const double ymin = -0.5, ymax = 0.5;
const double gas_gamma = 1.4;
const double gas_const = 1.0;
const int has_exact_sol = 0;
const double final_time = 80.0;

// Parameters for the initial condition
const double delta = 0.1;
const double M = 0.01;
const double R = 1.0e-3;

double eta(const double y)
{
   const double y1 = 9.0 / 32.0;
   const double y2 = 7.0 / 32.0;
   if (-y1 <= y && y < -y2)
      return 0.5 * (1.0 + sin(16.0 * M_PI * (y + 0.25)));
   else if (-y2 <= y && y < y2)
      return 1.0;
   else if (y2 <= y && y < y1)
      return 0.5 * (1.0 - sin(16.0 * M_PI * (y - 0.25)));
   else
      return 0.0;
}

void exactsol(const double t, const double x1, const double y1, double *Prim)
{
   PetscPrintf(PETSC_COMM_WORLD,"exactsol not implemented\n");
   abort();
}

void initcond(const double x, const double y, double *Prim)
{
   Prim[1] = M * (1.0 - 2.0 * eta(y));
   Prim[2] = delta * M * sin(2.0 * M_PI * x);

   Prim[0] = gas_gamma + R * (1.0 - 2.0 * eta(y));
   Prim[3] = 1.0;
}

void boundary_value(const double t, const double x, const double y, double *Con)
{
   PetscPrintf(PETSC_COMM_WORLD,"boundary_value not implemented\n");
   abort();
}

/* ---------------------------------------------------------------------
 * 
 *  Modified by the G8 group UU 2017
 * 
 *----------------------------------------------------------------------*/

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <vector> // This is for the std::vector
#include <cmath>
//#include <deal.II/lac/linear_operator.h>

#include <deal.II/lac/sparse_direct.h>        //direct solver
//#include <lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>//Trilinos precond
#include <deal.II/base/timer.h>               // Gives the comp_timer
#include <deal.II/numerics/data_out.h>        //cond OStream

#include "mgPrecondition.h"

using namespace dealii;
class Laplace
{
public:
  Laplace ();
  void run();
private:
  void solve ();
  void mgGLT();
  void solve_tril_amg();
  void clean();

  /* ========= Methods needed for initializing GLT ===================================================================== */
  void inputFile_supplied(unsigned int, unsigned int,std::string,SparsityPattern &sp, SparseMatrix<double> &sm);
  void writeToFile(SparseMatrix<double> &matrix,std::string filename);
  void kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
    SparsityPattern &sp, SparseMatrix <double> &M);
  void kronProd_vector(Vector<double> &A, Vector<double> &B,
    SparsityPattern &sp, SparseMatrix <double> &M);
  void spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa);
  void transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M);
  void prol_const(double n, SparsityPattern &spP, SparseMatrix<double> &P);
  void transMultMult(SparseMatrix<double> &P, SparseMatrix<double> &b, SparsityPattern &sp, SparseMatrix<double> &sM);
  double vectorProd(Vector<int> v1, Vector<int> v2);
  Vector<int> factor(const int N);
  Vector<int> unique(Vector<int> factor);
  Vector<int> accumVector(Vector<int> v);
/*======================================================================================================================*/
 
 int n_ref;
 int system_size;

 SparsityPattern *spP;
 SparsityPattern *spB;

 SparseMatrix<double> *BB;
 SparseMatrix<double> *PP;

// SparsityPattern      sparsity_pattern;
// SparseMatrix<double> system_matrix;

 Vector<double>       solution; // numeric solution
 Vector<double>       system_rhs;
 Vector<double>       solution_true;  //true solution

  // Measurements parameters
 ConditionalOStream   pcout;
 TimerOutput computing_timer;

 std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;

};
Laplace::Laplace ()
:
pcout(std::cout),
computing_timer (pcout,
 TimerOutput::summary,
 TimerOutput::wall_times)
{

/* Here you choose the size of the system;
	Preset is 6-refinements system
*/

  // system with 160mb AFin
  n_ref = 6;
  system_size = 274625;

  // system with 1.4 GB AFin
  //n_ref = 7;
  //system_size = 2146689;

  // systen with 12 GB AFin
  //n_ref = 8;
  //system_size = 16974593;

  spP = new SparsityPattern [n_ref];
  spB = new SparsityPattern [n_ref+1];

  PP = new SparseMatrix<double> [n_ref];
  BB = new SparseMatrix<double> [n_ref+1]; // BB[0] = AFin
  inputFile_supplied(system_size,system_size,"AFin.txt",spB[0],BB[0]);

  solution_true.reinit(system_size);
  solution.reinit(system_size);
  system_rhs.reinit(system_size);
  solution_true = 1;
  BB[0].vmult(system_rhs,solution_true);
}


/* Here you choose which version to run:
	Direct Solver
	AMG
	GLT-MG
*/

void Laplace::run(){

  // The Direct solver
  //solution = 0;
  //computing_timer.enter_section("total No Pre");
  //solve();
  //computing_timer.exit_section("total No Pre");

  /*
  // AMG
  solution = 0;
  computing_timer.enter_section("total Tril amg Pre");
  solve_tril_amg();
  computing_timer.exit_section("total Tril amg Pre");
  */

  //GLT-MG
  solution = 0;
  computing_timer.enter_section("total mgGLT Pre");
  mgGLT();
  computing_timer.exit_section("total mgGLT Pre");

  clean();
  
}

/* Clean up after you, you filthy animal */
void Laplace::clean(){
      // Freeing memory allocation
  delete[] PP;
  delete[] BB;
  delete[] spP;
  delete[] spB;
}

void Laplace::solve ()
{
      /* Original Laplace::solve */ 
  SolverControl           solver_control (system_size, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (BB[0], solution, system_rhs,
    PreconditionIdentity());
}


/* This method is to test initializing the GLT.
  Its content will later be included in solve()
  */
void Laplace::mgGLT(){
  computing_timer.enter_section("initialization mgGLT Pre");
  int N = BB[0].m(); // = size
  Vector<int> NN = factor(N);
  Vector<int> nu = unique(NN);
  Vector<int> reps = accumVector(NN);
  double n = vectorProd(reps,nu);
  int level = 0;    
  //BB{i} = AFin;
  //PP{i} = 0;
  double n1 = n;
  while(n1>=3){
      // P = (1/n1)*prol([. . .][ . . .])
      //PP{i} = P
    prol_const(n1,spP[level],PP[level]);
    // B = P'*BB{i}*P
    // BB{i+1} = B
    transMultMult(PP[level],BB[level], spB[level+1], BB[level+1]);    
    n1 = (n1+1)/2;
    level++;
  }
  std::cout<<" this is n: "<<n<<std::endl;
  mgPrecondition mg(BB, PP,system_rhs,n);
  computing_timer.exit_section("initialization mgGLT Pre");

  computing_timer.enter_section("solve mgGLT Pre");
  SolverControl           solver_control (system_size, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (BB[0], solution, system_rhs,mg);
  computing_timer.exit_section("solve mgGLT Pre");
  std::cout<<" solver with mgP finished in " << solver_control.last_step()<<std::endl;
  
  /* - test error norm - */
  Vector<double> diff(system_size);
  diff+=solution_true;
  diff-=solution;
  std::cout<<" norm of difference between sol_true and sol: "<< diff.norm_sqr()<<std::endl;
  
}

void Laplace::solve_tril_amg(){
          /*Defing the AMG Preconditioner, pulled from Anders code */
  computing_timer.enter_section("initialization Tril amg Pre");  
  Amg_preconditioner.reset();
  Amg_preconditioner = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> (new TrilinosWrappers::PreconditionAMG());
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.elliptic = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;
  //Amg_data.smoother_type="block Gauss-Seidel";
  Amg_data.smoother_type="Jacobi";
  Amg_preconditioner->initialize(BB[0], Amg_data);
  computing_timer.exit_section("initialization Tril amg Pre");

  computing_timer.enter_section("solve Tril amg Pre");
  SolverControl           solver_control (system_size, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (BB[0], solution, system_rhs,*Amg_preconditioner);
  std::cout<< "CG iterations with AMG preconditioner:"<<solver_control.last_step()<<std::endl;
  Amg_preconditioner->clear();
  computing_timer.exit_section("solve Tril amg Pre");

}



/*========================================== Methods added for GLT_init ==============================================================================*/


void Laplace::inputFile_supplied(unsigned int size1, unsigned int size2,std::string filename, 
  SparsityPattern &sp, SparseMatrix<double> &sm){

  DynamicSparsityPattern dsp(size1,size2);
  std::ifstream input_file(filename);
  unsigned int i,j;
  double val;

  if(!input_file){
    std::cout << "While opening file, error encounterde"<<std::endl;
  }
  else{
    //std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file>>i>>j>>val)//(!input_file.eof())
  {
    dsp.add(i-1,j-1);
  }
  input_file.close();
  dsp.compress();
  sp.copy_from(dsp);
  sm.reinit(sp);

  std::ifstream input_file_again(filename);
  if(!input_file_again){
    std::cout << "While opening a file an error is encountered" << std::endl;
  }
  else{
    //std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file_again>>i>>j>>val)//(!input_file.eof())
  {
    sm.add(i-1,j-1,val);
  }
  input_file_again.close();
}

void Laplace::writeToFile(SparseMatrix<double> &matrix,std::string filename){
  // filename ex "matrix.txt"
  std::ofstream out_system (filename);
  matrix.print_formatted(out_system,3,true,0," ",1); 
}

/* Vector A is percieved to be a 1xn vector
and Vector B is percieved to be a mx1 vector!
        resulting Matrix M is a mxn matrix
*/
void Laplace::kronProd_vector(Vector<double> &A, Vector<double> &B,
  SparsityPattern &sp, SparseMatrix <double> &M){
  const int m = B.size();
  const int n = A.size();
  DynamicSparsityPattern dsp(m,n);
  Vector<double>::iterator itA = A.begin();
  Vector<double>::iterator itB = B.begin();
  Vector<double>::iterator endA = A.end();
  Vector<double>::iterator endB = B.end();
  int i=0,j=0;
  double val,valA,valB;
  for(; itA!=endA; itA++){
    for(; itB!=endB; itB++){
      dsp.add(j,i);
      j++;
    }
    itB = B.begin();
    j=0;
    i++;
  }
  dsp.compress();
  sp.copy_from(dsp);
  M.reinit(sp);
  itA = A.begin();
  i=0; j=0;
  for(; itA!=endA; itA++){
    valA = *itA;
    for(; itB!=endB; itB++){
      valB = *itB;
      val = valA*valB;
      M.add(j,i,val);
      j++;
    }
    itB = B.begin();
    j=0;
    i++;
  }
}

/*
        This is the kronecker product of 2 sparse matrices
*/
void Laplace::kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
  SparsityPattern &sp, SparseMatrix <double> &M){
  const int nA = A.n(); //nCols
  const int nB = B.n();
  const int mA = A.m(); //nRows
  const int mB = B.m();
  const int n = nA*nB;
  const int m = mA*mB;
  DynamicSparsityPattern dsp(m,n);
  SparseMatrix<double>::iterator itA = A.begin();
  SparseMatrix<double>::iterator endA = A.end();
  SparseMatrix<double>::iterator itB = B.begin(); // I fixed it ;P
  SparseMatrix<double>::iterator endB = B.end();
  int rowA, colA, rowB, colB;
  int i,j;
  double val,valA,valB;
  for(; itA!=endA; itA++){
    rowA = itA->row();
    colA = itA-> column();
    for(; itB!=endB; itB++){
      rowB = itB->row();
      colB = itB-> column();
      i = rowA*mB+rowB;
      j = colA*nB+colB;
      dsp.add(i,j);
    }
    itB = B.begin();
  }
  dsp.compress();
      // Should we put a sp.clear?
  sp.copy_from(dsp);
  M.reinit(sp);
  itA = A.begin();
  for(; itA!=endA; itA++){
    rowA = itA->row();
    colA = itA-> column();
    valA = itA->value();
    for(; itB!=endB; itB++){
      rowB = itB->row();
      colB = itB-> column();
      valB = itB->value();
      i = rowA*mB+rowB;
      j = colA*nB+colB;
      val = valA*valB;
      M.add(i,j,val);
    }
    itB = B.begin();
  }
}
/* Here we implement the prol([2 1 1],[0 -1 1],n)
        dd = [0 -1 1]
        PP = [2 1 1]
        performs the operation:
        aa = spdiags(kron(PP,one(n,1),dd,n,n));
*/
void Laplace::spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa){
  Vector<double> dd;
  dd.reinit(3);
  dd[0] = 0; dd[1] = -1; dd[2] = 1;
  Vector<double> PP;
  PP.reinit(3);
  PP[0] = 2; PP[1] = 1; PP[2] = 1; // Obs!!!
  int N = (int)n;
  /* Make the sm1 = kron(PP,ones(n,1)) matrix! :D */
  Vector<double> ones;
  ones.reinit(N);
  ones = 1;
  SparsityPattern sp1;
  SparseMatrix<double> sm1;
  kronProd_vector(PP,ones,sp1,sm1);
        /* I am hardcoding the diagonals. i.e. I dont use the dd matrix. Sorry ;)*/
  DynamicSparsityPattern dsp(N,N);
        // ! ! ! ! !  Possible problem if the matrix is shorter than N !! ! ! !
  for(int i=0; i<N-1; i++){
    dsp.add(i,i);
    dsp.add(i+1,i);
    dsp.add(i,i+1);
  }
  dsp.add(N-1,N-1);
  dsp.compress();
  spaa.copy_from(dsp);
  aa.reinit(spaa);
  for(int i=0; i<N-1; i++){
    aa.add(i,i,sm1(i,0));   // Column 0 on diag 0
    aa.add(i+1,i,sm1(i+1,1)); // Column 1 on diag -1
    aa.add(i,i+1,sm1(i,2));   // Column 2 on diag 1
  }
  aa.add(N-1,N-1,sm1(N-1,0));
}

/*  Here we make a transpose of matrix A which can be any type of a matrix
        output sp and M */
void Laplace::transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M){
  const int n = A.m();
  const int m = A.n();
  DynamicSparsityPattern dsp(m,n);
  SparseMatrix<double>::iterator itA = A.begin();
  SparseMatrix<double>::iterator endA = A.end();
  double value;
  int rowM=0,colM=0;
  for(;itA!=endA;++itA){
    rowM=itA->column();
    colM=itA->row();
    dsp.add(rowM,colM);
  }
  dsp.compress();
  sp.copy_from(dsp);
  M.reinit(sp);
  itA = A.begin();
  for(;itA!=endA;itA++){
    rowM=itA->column();
    colM=itA->row();
    value = itA->value();
    M.add(rowM,colM,value);
  }
}

/* This method performs the prod(nu.^(resp(nu)./3))
        action given in the GLTmg*/
double Laplace::vectorProd(Vector<int> reps, Vector<int> nu){
  double n = 1;
    // the reps(nu) action
  Vector<double> repsNu;  //Used for development
  repsNu.reinit(nu.size());
  Vector<int>::iterator nu_iter = nu.begin();
  Vector<int>::iterator nu_ender = nu.end();
  int i =  0;
  for(; nu_iter!=nu_ender; nu_iter++){
    repsNu(i) = (double)(reps(*nu_iter-1))/3;
    n = n*pow(nu(i),repsNu(i));
    i++;
  }
  return n;
}

/* Return a vector containing the primenumbers of N
        Might Implement Sieve at a later point    */
Vector<int> Laplace::factor(int N){
  int index = 0;
  const int maxSize = 10;
  int factors[maxSize];
  while(N%2==0){
                //Add 2
    factors[index] = 2;
    index++;
    N=N/2;
  }
  for(int i=3; i<=sqrt(N); i=i+2){
    while(N%i==0){
                        //add i
      factors[index] = i;
      index++;
      N=N/i;
    }
  }
  if(N>2){
                //add (N)
    factors[index] = N;
    index++;
  }
  Vector<int> result;
  result.reinit(index);
  Vector<int>::iterator iter = result.begin();
  Vector<int>::iterator ender = result.end();
  int i = 0;
  for(; iter!=ender; iter++){
    *iter = factors[i];
    i++;
  }
  return result;
}

/* removes extra numbers from factors */
Vector<int> Laplace::unique(Vector<int> factor){
  if(factor.size()==1){
    return factor;
  } else{
    Vector<int> unique;
    const int maxSize = 10;
    int tmp[maxSize];
    int i = 0;
    Vector<int>::iterator iter = factor.begin();
    Vector<int>::iterator ender = factor.end();
    tmp[i] = *iter;
    iter++;
    for(; iter!=ender; iter++){
      if(*iter!=tmp[i]){
        i++;
        tmp[i] = *iter;
      }
    }
    unique.reinit(i+1);
    iter = unique.begin();
    ender = unique.end();
    for(i=0;iter!=ender; iter++){
      *iter = tmp[i];
      i++;
    }
    return unique;
  }
}

/* Returns an accumVector. This method assumes v is sorted!
        Note! The GLTmg has *ugh* Matlab indexing !
        Returnes # of Values of pos int on index i !
*/
Vector<int> Laplace::accumVector(Vector<int> v){
  Vector<int>::iterator iter = v.begin();
  Vector<int>::iterator ender = v.end();
  Vector<int> res;
  int res_size = *(ender-1);
  res.reinit(res_size); //prolong so you can access the last element!
  res = 0;
  for(; iter!=ender; iter++){
    res(*iter-1)++;
  }
  return res;
}

/* This Method performs the B = P'*BB{i}*P operation, where BB is a std::vector
    and a member of mgPrecondtion. The method returns the const matrix P */
void Laplace::transMultMult(SparseMatrix<double> &P, SparseMatrix<double> &b, SparsityPattern &sp,
  SparseMatrix<double> &sm){
  SparsityPattern spTemp;
  SparseMatrix<double> temp;
  
  DynamicSparsityPattern dspTemp(0),dspB(0);
  spTemp.copy_from(dspTemp);
  temp.reinit(spTemp);

  b.mmult(temp, P, Vector<double>(), true);

  sp.copy_from(dspB);
  sm.reinit(sp);
  P.Tmmult(sm,temp, Vector<double>(),true);
}

/* This is an implementation of the algorithm
        (1/n)*prol([2 1 1], [0 -1 1],n)
        given in the gltmg_test matlab code

        This code is memory-ineffective and should include destructors

        Method tested and seems to work.... Finaly....

  We change the prol method such that it can return a const SparseMatrix :)
 */
void Laplace::prol_const(double n, SparsityPattern &spP,SparseMatrix<double> &P){
  // aa = spdiags(kron(PP,ones),dd,n,n)
  SparsityPattern spaa;
  SparseMatrix<double> aa;
  spdiags(n,spaa,aa);
  // smP = kron(aa,kron(aa,aa))
  SparsityPattern spTemp;
  SparseMatrix<double> smTemp;
  kronProd(aa,aa,spTemp,smTemp);
  SparsityPattern spP2;
  SparseMatrix<double> smP;
  kronProd(aa,smTemp,spP2,smP);
  // Create smH = speye(n) then remove every other row
  // smH = smH(1:2:n,:)
  SparsityPattern spH2;
  SparseMatrix<double> smH;
  int N1 = (int)n;
  int N2 = N1/2+N1%2;
  DynamicSparsityPattern dspH(N2,N1);
  for(int i=0; i<N2; i++){
    dspH.add(i,2*i);
  }
  dspH.compress();
  spH2.copy_from(dspH);
  smH.reinit(spH2);
  for(int i=0; i<N2; i++){
    smH.add(i,2*i,1);
  }
  // H = kron(smH,kron(smH,smH))
  SparsityPattern sp_dummyH;
  SparseMatrix<double> dummyH; 
  SparsityPattern spH;
  SparseMatrix<double> H;
  kronProd(smH,smH,sp_dummyH,dummyH);
  kronProd(smH,dummyH,spH,H);
  // P = smP*H'; P = (1/n)*P
  SparsityPattern spTranspH;
  SparseMatrix<double> transpH;
  //SparseMatrix<double> P;
  transp(H,spTranspH,transpH);
  DynamicSparsityPattern dspP(0);
  spP.copy_from(dspP);
  P.reinit(spP);
  smP.mmult(P,transpH,Vector<double>(),true);
  //P = (1/n).P
  P*=(1/n);
}

/*=====================================================================================================================================================*/

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  //deallog.depth_console (2);
  //Laplace laplace_problem;
  //laplace_problem.run ();
  
  Laplace test;
  test.run();

  return 0;
}

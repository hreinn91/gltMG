
/*-----------------------------------
- GLT-MG; implementation in deal.ii
- by Hreinn Juliusson aut.17
-------------------------------------*/

#ifndef mgPreconditionition_H
#define mgPreconditionition_H

#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/base/config.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/identity_matrix.h> // IdentityMatrix inclusion :)
#include <vector> // This is for the std::vector
#include <tgmath.h> //floor


DEAL_II_NAMESPACE_OPEN
//using namespace mgPrecondition;

class mgPrecondition: public Subscriptor{

public:
	mgPrecondition(SparseMatrix<double> *bb, SparseMatrix<double> *pp,
   Vector<double> &vector, int n_levels);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;
    void glt_setup(const Vector<double> &b, Vector<double> &dst, int n, int level) const; // my recursion
private:
    SparseMatrix<double> *BB;
    SparseMatrix<double> *PP;
    Vector<double> rhs;
    int size;
    int n;

    double tol;     //tolerance
    int max_iterations; //max number of MG cycles
};

mgPrecondition::mgPrecondition(SparseMatrix<double> *bb, SparseMatrix<double> *pp,
   Vector<double> &vector, int n_levels){
    tol = 0.00000001; //1e-8;
    max_iterations = 3; //number of MG cycles
    PP = pp;
    BB = bb;
    rhs = vector;
    size = BB[0].m();
    n = n_levels;
}

/* ===================== The functions needed to perform GLTmg ==============================================================*/

void mgPrecondition::vmult(Vector<double> &dst, const Vector<double> &src) const {
    //dst.reinit(size); // x = zeros(size(b))
    dst = 0;
    // r = rhs-Ax; x=0 -> r=rhs;
    Vector<double> r(src); // Copy
    double tol_main = tol*src.norm_sqr();
    
    int i = 0;
    while(r.norm_sqr()>tol_main && i<max_iterations){
        // recursion
        glt_setup(src, dst, n,0);
        // r = rhs-BB[0]*x
        BB[0].residual(r,dst,src);
        i++;
    }
  //  int n_iter = i-1;     // Number of iterations
}

/* include x in the function statement !
   x = mgGLT_setup(A,b,x,n,level)
 */
void mgPrecondition::glt_setup(const Vector<double> &b, Vector<double> &dst, int n, int level) const{
    //int size = b.size();
    if(n<5){                //5<=
        // calc the inverse
        // A/b = x
        SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(BB[level]);
        direct_solver.vmult(dst,b);
    } else{
        // P = PP[level]
        // x = presmoth(A,b,x)
        BB[level].Jacobi_step(dst,b,1);
        // r = b-A*x
        int pp_m = b.size();
        Vector<double> r(pp_m); //r(b.size());
        BB[level].residual(r,dst,b);
        // d = P'*r
        int pp_n = PP[level].n();
        Vector<double> d(pp_n);
        PP[level].Tvmult(d,r);

        // k = floor((n+1)*0.5)
        double k = std::floor((n+1)*0.5);
        // e = zeros(size(d))
        Vector<double> e(pp_n);
        // for j= 1 (V cycle)
        // B = BB[level+1]
        // e = mgGLT(B,d,e,k,level+1)
        glt_setup(d,e,k,level+1);
        // g = P*e
        Vector<double> g(pp_m);
        PP[level].vmult(g,e);
        // x = x + g
        dst += g;
        // x = postsmooth(A,b,x)
        double damp = 2./3;
        BB[level].Jacobi_step(dst,b,damp);

    }

}


/*==========================================================================================================================*/


DEAL_II_NAMESPACE_CLOSE
#endif



/*

 __
 | |    The code dump
 | |
 | |
 | |
 | |		   _____
 | |		  |		|	   _____
 | | Hreinn	  |		|	  |     |
_| |__________|		|_____|     |___________



Test/Help functions! 

void mgPrecondition::printMatrix(){
    //Why do we use arrow here? -> = (*).
    //system_matrix->print_formatted(std::cout,1,true,0," ",1);
}
void mgPrecondition::printVector(){
    //rhs->print(std::cout,1,true,true);
}


*/

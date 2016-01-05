/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef QuadratureRule_h
#define QuadratureRule_h

#include <vector>
#include <Teuchos_SerialDenseVector.hpp>

namespace sierra{
namespace naluUnit{

  std::pair<Teuchos::SerialDenseVector<int,double>, Teuchos::SerialDenseVector<int,double>>
  jacobi_recursion_coefficients(
    const double alpha,
    const double beta,
    const int order);

  std::pair<std::vector<double>, std::vector<double>>
  gauss_legendre_rule(int order);


  std::pair<Teuchos::SerialDenseVector<int,double>, Teuchos::SerialDenseVector<int,double>>
  coefficients_for_lobatto(
    int order,
    double xl1,
    double xl2);


  std::pair<std::vector<double>, std::vector<double>>
  gauss_lobatto_legendre_rule(int order, double xleft = -1.0, double xright = +1.0);

} // namespace nalu
} // namespace Sierra

#endif

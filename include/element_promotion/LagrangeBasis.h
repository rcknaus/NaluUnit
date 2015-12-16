/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductBasis_h
#define TensorProductBasis_h

#include <vector>
#include <map>
#include <string>
#include <cstdlib>

namespace sierra{
namespace naluUnit{

class LagrangeBasis
{
public:
  LagrangeBasis(
    std::vector<std::vector<unsigned>>  indicesMap,
    const std::vector<double>& nodeLocs
  );

  virtual ~LagrangeBasis() {};


  std::vector<double> eval_basis_weights(unsigned dimension, const std::vector<double>& intgLoc);

  std::vector<double> eval_deriv_weights(
    unsigned dimension,
    const std::vector<double>& intgLoc);

  void set_lagrange_weights();

  double tensor_lagrange_derivative(
    unsigned dimension,
    const double* x,
    const unsigned* nodes,
    unsigned derivativeDirection
  );
  double tensor_lagrange_interpolant(unsigned dimension, const double* x, const unsigned* nodes);
  double lagrange_1D(double x, unsigned nodeNumber);
  double lagrange_deriv_1D(double x, unsigned nodeNumber);

  std::vector<std::vector<unsigned>> indicesMap_;
  unsigned numNodes1D_;
  std::vector<double> nodeLocs_;
  std::vector<double> lagrangeWeights_;
};


} // namespace nalu
} // namespace Sierra

#endif

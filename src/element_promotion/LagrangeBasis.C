/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include "../../include/element_promotion/LagrangeBasis.h"

#include <iostream>
#include <cmath>
#include <limits>

namespace sierra{
namespace naluUnit{

LagrangeBasis::LagrangeBasis(
  std::vector<std::vector<unsigned>>  indicesMap,
  const std::vector<double>& nodeLocs)
  :  indicesMap_(std::move(indicesMap)),
     numNodes1D_(nodeLocs.size()),
     nodeLocs_(nodeLocs)
{
  set_lagrange_weights();
}
//--------------------------------------------------------------------------
void
LagrangeBasis::set_lagrange_weights()
{
  lagrangeWeights_.assign(numNodes1D_,1.0);
  for (unsigned i = 0; i < numNodes1D_; ++i) {
    for (unsigned j = 0; j < numNodes1D_; ++j) {
      if ( i != j ) {
        lagrangeWeights_[i] *= (nodeLocs_[i]-nodeLocs_[j]);
      }
    }
    lagrangeWeights_[i] = 1.0 / lagrangeWeights_[i];
  }
}
//--------------------------------------------------------------------------
std::vector<double>
LagrangeBasis::eval_basis_weights(
  unsigned dimension,
  const std::vector<double>& intgLoc)
{
  auto numIps = intgLoc.size() / dimension;
  auto numNodes = std::pow(numNodes1D_, dimension);
  std::vector<double> interpolationWeights(numIps*numNodes);

  for (unsigned ip = 0; ip < numIps; ++ip) {
    unsigned scalar_ip_offset = ip*numNodes;
    for (unsigned nodeNumber = 0; nodeNumber < numNodes; ++nodeNumber) {
      unsigned scalar_offset = scalar_ip_offset+nodeNumber;
      unsigned vector_offset = ip * dimension;
      interpolationWeights[scalar_offset]
          = tensor_lagrange_interpolant( dimension,
                                        &intgLoc[vector_offset],
                                         indicesMap_[nodeNumber].data() );
    }
  }
  return interpolationWeights;
}
//--------------------------------------------------------------------------
std::vector<double>
LagrangeBasis::eval_deriv_weights(
  unsigned dimension,
  const std::vector<double>& intgLoc)
{
  auto numIps = intgLoc.size()/dimension;
  auto numNodes = std::pow(numNodes1D_,dimension);
  std::vector<double> derivWeights(numIps * numNodes * dimension);

  unsigned derivIndex = 0;
  for (unsigned ip = 0; ip < numIps; ++ip) {
    for (unsigned nodeNumber = 0; nodeNumber < numNodes; ++nodeNumber) {
      unsigned vector_offset = ip*dimension;
      for (unsigned derivDirection = 0; derivDirection < dimension; ++derivDirection) {
        derivWeights[derivIndex]
            = tensor_lagrange_derivative( dimension,
                                         &intgLoc[vector_offset],
                                          indicesMap_[nodeNumber].data(),
                                          derivDirection );
        ++derivIndex;
      }
    }
  }
  return derivWeights;
}
//--------------------------------------------------------------------------
double
LagrangeBasis::tensor_lagrange_interpolant(unsigned dimension, const double* x, const unsigned* nodes)
{
  double interpolant_weight = 1.0;
  for (unsigned j = 0; j < dimension; ++j) {
    interpolant_weight *= lagrange_1D(x[j], nodes[j]);
  }
  return interpolant_weight;
}
//--------------------------------------------------------------------------
double
LagrangeBasis::tensor_lagrange_derivative(unsigned dimension,
  const double* x,
  const unsigned* nodes,
  unsigned derivativeDirection)
{
  double derivativeWeight = 1.0;
  for (unsigned j = 0; j < dimension; ++j) {
    if (j == derivativeDirection) {
      derivativeWeight *= lagrange_deriv_1D(x[j], nodes[j]);
    }
    else {
      derivativeWeight *= lagrange_1D(x[j], nodes[j]);
    }
  }
  return derivativeWeight;
}
//--------------------------------------------------------------------------
double
LagrangeBasis::lagrange_1D(double x, unsigned nodeNumber)
{
  double numerator = 1.0;
  for (unsigned j = 0; j < numNodes1D_; ++j) {
    if (j != nodeNumber) {
      numerator *= (x - nodeLocs_[j]);
    }
  }
  return (numerator * lagrangeWeights_[nodeNumber]);
}
//--------------------------------------------------------------------------
double
LagrangeBasis::lagrange_deriv_1D(double x, unsigned nodeNumber)
{
  double outer = 0.0;
  for (unsigned j = 0; j < numNodes1D_; ++j) {
    if (j != nodeNumber) {
      double inner = 1.0;
      for (unsigned i = 0; i < numNodes1D_; ++i) {
        if (i != j && i != nodeNumber) {
          inner *= (x - nodeLocs_[i]);
        }
      }
      outer += inner;
    }
  }
  return (outer * lagrangeWeights_[nodeNumber]);
}

}  // namespace naluUnit
} // namespace sierra

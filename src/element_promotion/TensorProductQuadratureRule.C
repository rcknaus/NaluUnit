/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/TensorProductQuadratureRule.h>
#include <cmath>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

TensorProductQuadratureRule::TensorProductQuadratureRule(
  std::string  /*type*/,
  int order,
  std::vector<double>& scsLocs)
{
  scsEndLoc_.resize(scsLocs.size()+2);
  scsEndLoc_[0] = -1.0;

  for (unsigned j = 0; j < scsLocs.size();++j) {
    scsEndLoc_[j+1] = scsLocs[j];
  }
  scsEndLoc_[scsLocs.size()+1] = +1.0;


  switch (order) {
    case 1:
    {
      abscissae_ = { 0.0 };
      weights_ = { 1.0 };
      break;
    }
    case 2: case 3:
    {
      abscissae_ = { -std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };
      weights_ = { 0.5, 0.5 };
      break;
    }
    case 4: case 5:
    {
      abscissae_ = { -std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0) };
      weights_ = { 5.0/18.0, 4.0/9.0,  5.0/18.0 };
      break;
    }
    case 6: case 7:
    {
      abscissae_ = {
          -std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0)),
          -std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
          +std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
          +std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0))
      };

      weights_ = {
          (18.0-std::sqrt(30.0))/72.0,
          (18.0+std::sqrt(30.0))/72.0,
          (18.0+std::sqrt(30.0))/72.0,
          (18.0-std::sqrt(30.0))/72.0
      };
      break;
    }
    default: {
      throw std::runtime_error("Quadrature rule not implemented");
    }
  }
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::isoparametric_mapping(
  const double b,
  const double a,
  const double xi) const
{
  return xi*(b-a)/2.0 +(a+b)/2.0;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{

  double location1D =
      isoparametric_mapping( scsEndLoc_[nodeOrdinal+1],
                             scsEndLoc_[nodeOrdinal],
                             abscissae_[gaussPointOrdinal] );
   return location1D;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::tensor_product_weight(
  int s1Node, int s2Node,
  int s1Ip, int s2Ip) const
{
  //surface integration
  const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
  const double isoparametricArea = Ls1 * Ls2;
  const double weight = isoparametricArea * weights_[s1Ip] * weights_[s2Ip];

  return weight;
}
//--------------------------------------------------------------------------
double
TensorProductQuadratureRule::tensor_product_weight(int s1Node, int s1Ip) const
{
  const double isoparametricLength = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
  const double weight = isoparametricLength * weights_[s1Ip];

  return weight;
}

}  // namespace naluUnit
} // namespace sierra

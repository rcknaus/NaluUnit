/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef QuadratureRule_h
#define QuadratureRule_h

#include <vector>

namespace sierra{
namespace naluUnit{

  // <abscissae, weights>
  std::pair<std::vector<double>, std::vector<double>>
  gauss_legendre_rule(int order);

  // <abscissae, weights>
  std::pair<std::vector<double>, std::vector<double>>
  gauss_lobatto_legendre_rule(int order, double xleft = -1.0, double xright = +1.0);

  // <abscissae, weights>
  std::pair<std::vector<double>, std::vector<double>>
  SGL_quadrature_rule(int order, std::vector<double> scsEndLocations);

} // namespace nalu
} // namespace Sierra

#endif

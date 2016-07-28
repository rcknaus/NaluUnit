/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <element_promotion/QuadratureRule.h>
#include <KokkosInterface.h>
#include <Teuchos_LAPACK.hpp>

namespace sierra {
namespace naluUnit {
namespace CoefficientMatrices {

/* Computes 1D coefficient matrices (e.g. for the derivative) for CVFEM */


// Some Lagrange interpolant functions returning Kokkos views
namespace LagrangeTools {

template<unsigned poly_order>
typename LineViews<poly_order>::nodal_scalar_array
barycentric_weights(const double* nodeLocs)
{
  typename LineViews<poly_order>::nodal_scalar_array lagrangeWeights("barycentric weights");
  const unsigned nodes1D = poly_order+1;
  for (unsigned i = 0; i < nodes1D; ++i) {
    lagrangeWeights(i) = 1.0;
  }
  for (unsigned i = 0; i < nodes1D ; ++i) {
    for (unsigned j = 0; j < nodes1D ; ++j) {
      if ( i != j ) {
        lagrangeWeights(i) *= (nodeLocs[i]-nodeLocs[j]);
      }
    }
    lagrangeWeights(i) = 1.0 / lagrangeWeights(i);
  }
  return lagrangeWeights;
}
//--------------------------------------------------------------------------
template < unsigned poly_order > double
lagrange_interp_1D(const double* nodeLocs, double x, unsigned i) {
  double numerator = 1.0;
  for (unsigned k = 0; k < poly_order+1; ++k) {
    if (i != k) {
      numerator *= (x - nodeLocs[k]);
    }
  }
  return numerator;
}
//--------------------------------------------------------------------------
template < unsigned poly_order > double
lagrange_deriv_1D(const double* nodeLocs, double x, unsigned i)
{
  double outer = 0.0;
  for (unsigned k = 0; k < poly_order+1; ++k) {
    if (k != i) {
      double inner = 1.0;
      for (unsigned r = 0; r < poly_order+1; ++r) {
        if (r != k && r != i) {
          inner *= (x - nodeLocs[r]);
        }
      }
      outer += inner;
    }
  }
  return outer;
}

} // end namespace LagrangeTools
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientMatrixViews<poly_order>::nodal_matrix_array
nodal_integration_weights(
  const double* nodeLocs,
  const double* scsLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  constexpr unsigned nodesPerElement = (poly_order+1)*(poly_order+1);

  typename QuadViews<poly_order>::nodal_scalar_array weightLHS("vandermonde matrix");
  for (unsigned j = 0; j < nodes1D; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      weightLHS(j,i) = std::pow(nodeLocs[j], i);
    }
  }

  typename QuadViews<poly_order>::nodal_scalar_array weights("nodal integration weighting for each scv");
  // each node has a separate RHS
  for (unsigned i = 0; i < nodes1D; ++i) {
    weights(0,i) = (std::pow(scsLocs[0], i + 1) - std::pow(-1.0, i + 1)) / (i + 1.0);
  }

  for (unsigned j = 1; j < nodes1D-1; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      weights(j,i) = (std::pow(scsLocs[j], i + 1) - std::pow(scsLocs[j-1], i + 1)) / (i + 1.0);
    }
  }

  for (unsigned i = 0; i < nodes1D; ++i) {
    weights(poly_order,i) = (std::pow(+1.0, i + 1) - std::pow(scsLocs[poly_order-1], i + 1)) / (i + 1.0);
  }

  int info = 1;
  int ipiv[nodesPerElement];
  Teuchos::LAPACK<int, double>().GESV(nodes1D, nodes1D,
    weightLHS.data(), nodes1D,
    ipiv,
    weights.data(), nodes1D,
    &info
  );
  ThrowRequire(info == 0);

  return weights;
}
//--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientMatrixViews<poly_order>::scs_matrix_array
scs_interpolation_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array scsInterp("subcontrol surface interpolation matrix");

  auto lagrangeWeights = LagrangeTools::barycentric_weights<poly_order>(nodeLocs);
  for (unsigned j = 0; j < poly_order; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      scsInterp(j,i) = LagrangeTools::lagrange_interp_1D<poly_order>(nodeLocs,scsLocs[j],i) * lagrangeWeights(i);
    }
  }
  return scsInterp;
}
//--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientMatrixViews<poly_order>::scs_matrix_array
scs_derivative_weights(
  const double* nodeLocs,
  const double* scsLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array scsDeriv("subcontrol surface derivative matrix");

  auto lagrangeWeights = LagrangeTools::barycentric_weights<poly_order>(nodeLocs);
  for (unsigned j = 0; j < poly_order; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      scsDeriv(j,i) = LagrangeTools::lagrange_deriv_1D<poly_order>(nodeLocs, scsLocs[j], i) * lagrangeWeights(i);
    }
  }
  return scsDeriv;
}
////--------------------------------------------------------------------------
template < unsigned poly_order >
typename CoefficientMatrixViews<poly_order>::nodal_matrix_array
nodal_derivative_weights(const double* nodeLocs)
{
  constexpr unsigned nodes1D = poly_order+1;
  typename QuadViews<poly_order>::nodal_scalar_array nodalDeriv("nodal derivative matrix");

  auto lagrangeWeights = LagrangeTools::barycentric_weights<poly_order>(nodeLocs);
  for (unsigned j = 0; j < nodes1D; ++j) {
    for (unsigned i = 0; i < nodes1D; ++i) {
      nodalDeriv(j,i) = LagrangeTools::lagrange_deriv_1D<poly_order>(nodeLocs, nodeLocs[j], i) * lagrangeWeights(i);
    }
  }
  return nodalDeriv;
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename QuadViews<poly_order>::nodal_scalar_array
nodal_integration_weights()
{
  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return nodal_integration_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename QuadViews<poly_order>::nodal_scalar_array
nodal_derivative_weights()
{
  std::vector<double> nodeLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);

  return nodal_derivative_weights<poly_order>(nodeLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientMatrixViews<poly_order>::scs_matrix_array
scs_derivative_weights()
{
  std::vector<double> nodeLocs;  std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return scs_derivative_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
//--------------------------------------------------------------------------
template <unsigned poly_order>
typename CoefficientMatrixViews<poly_order>::scs_matrix_array
scs_interpolation_weights()
{
  std::vector<double> nodeLocs; std::vector<double> scsLocs;
  std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
  std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

  return scs_interpolation_weights<poly_order>(nodeLocs.data(), scsLocs.data());
}
}

} // namespace naluUnit
} // namespace Sierra

#endif

/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderLaplacianQuad_h
#define HighOrderLaplacianQuad_h

#include <element_promotion/new_assembly/HighOrderOperators.h>
#include <KokkosInterface.h>

namespace sierra {
namespace naluUnit {

template < unsigned poly_order>
class HighOrderLaplacianQuad
{
//==========================================================================
// Class Definition
//==========================================================================
// HigherOrderLaplacianQuad - Computes elemental quantities for the assembly
// of the Poisson/Laplace equation for quadrilateral elements.
//
// TODO(rcknaus): Profiling in general
//==========================================================================

public:
  //--------------------------------------------------------------------------
  // Constructor with specifiable node locations / scs locations for high-order cvfem
  HighOrderLaplacianQuad(const double* nodeLocs, const double* scsLocs)
: ops(nodeLocs, scsLocs),
  grad_phi("scratch for the gradient of phi in parametric coordinates evaluated at the subcontrol-surfaces"),
  integrand("scratch space for integrand of surface fluxes"),
  flux("scratch space for the integrated surface fluxes")
  { };
  //--------------------------------------------------------------------------
  // Constructor for the usual high-order cvfem
  HighOrderLaplacianQuad()
: ops(),
  grad_phi("scratch for the gradient of phi in parametric coordinates evaluated at the subcontrol-surfaces"),
  integrand("scratch space for integrand of surface fluxes"),
  flux("scratch space for the integrated surface fluxes")
  { };
  //--------------------------------------------------------------------------
  void elemental_laplacian_matrix(
    const typename QuadViews<poly_order>::scs_tensor_array& metric,
    typename QuadViews<poly_order>::matrix_array& lhs)
  {
    /*
     * Computes the elemental lhs for the Laplacian operator given
     * the correct grid metrics, split into boundary and interior terms.
     */
    constexpr int nodes1D = poly_order + 1;

    // flux past constant yhat lines
    for (int n = 0; n < nodes1D; ++n) {
      // x- element boundary
      constexpr int m_minus = 0;
      for (int j = 0; j < nodes1D; ++j) {
        double orth = ops.nodalWeights(n, j) * metric(XH, XH, m_minus, j);
        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += ops.nodalWeights(n, k) * ops.nodalDeriv(k, j) * metric(XH, YH, m_minus, k);
        }

        for (int i = 0; i < nodes1D; ++i) {
          lhs(idx(n, m_minus), idx(j, i)) +=
              orth * ops.scsDeriv(m_minus, i) + non_orth * ops.scsInterp(m_minus, i);
        }
      }

      // interior flux
      for (int m = 1; m < nodes1D - 1; ++m) {
        for (int j = 0; j < nodes1D; ++j) {
          const double w = ops.nodalWeights(n, j);
          const double orthm1 = w * metric(XH, XH, m - 1, j);
          const double orthp0 = w * metric(XH, XH, m + 0, j);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < nodes1D; ++k) {
            const double wd = ops.nodalWeights(n, k) * ops.nodalDeriv(k, j);
            non_orthm1 += wd * metric(XH, YH, m - 1, k);
            non_orthp0 += wd * metric(XH, YH, m + 0, k);
          }

          for (int i = 0; i < nodes1D; ++i) {
            const double fm = orthm1 * ops.scsDeriv(m - 1, i) + non_orthm1 * ops.scsInterp(m - 1, i);
            const double fp = orthp0 * ops.scsDeriv(m + 0, i) + non_orthp0 * ops.scsInterp(m + 0, i);
            lhs(idx(n, m), idx(j, i)) += (fp - fm);
          }
        }
      }

      // x+ element boundary
      constexpr int m_plus = nodes1D - 1;
      for (int j = 0; j < nodes1D; ++j) {
        const double orth = ops.nodalWeights(n, j) * metric(XH, XH, m_plus - 1, j);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += ops.nodalWeights(n, k) * ops.nodalDeriv(k, j) * metric(XH, YH, m_plus - 1, k);
        }
        for (int i = 0; i < nodes1D; ++i) {
          lhs(idx(n, m_plus), idx(j, i)) -=
              orth * ops.scsDeriv(m_plus - 1, i) + non_orth * ops.scsInterp(m_plus - 1, i);
        }
      }
    }

    // flux past constant xhat lines
    for (int m = 0; m < nodes1D; ++m) {
      // y+ boundary
      constexpr int n_minus = 0;
      for (int i = 0; i < nodes1D; ++i) {
        const double orth = ops.nodalWeights(m, i) * metric(YH, YH, n_minus, i);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += ops.nodalWeights(m, k) * ops.nodalDeriv(k, i) * metric(YH, XH, n_minus, k);
        }
        for (int j = 0; j < nodes1D; ++j) {
          lhs(idx(n_minus, m), idx(j, i)) +=
              orth * ops.scsDeriv(n_minus, j) + non_orth * ops.scsInterp(n_minus, j);
        }
      }

      // interior flux
      for (int n = 1; n < nodes1D - 1; ++n) {
        for (int i = 0; i < nodes1D; ++i) {
          const double w = ops.nodalWeights(m, i);
          const double orthm1 = w * metric(YH, YH, n - 1, i);
          const double orthp0 = w * metric(YH, YH, n + 0, i);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < nodes1D; ++k) {
            const double wd = ops.nodalWeights(m, k) * ops.nodalDeriv(k, i);
            non_orthm1 += wd * metric(YH, XH, n - 1, k);
            non_orthp0 += wd * metric(YH, XH, n + 0, k);
          }

          for (int j = 0; j < nodes1D; ++j) {
            const double fm = orthm1 * ops.scsDeriv(n - 1, j) + non_orthm1 * ops.scsInterp(n - 1, j);
            const double fp = orthp0 * ops.scsDeriv(n + 0, j) + non_orthp0 * ops.scsInterp(n + 0, j);
            lhs(idx(n, m), idx(j, i)) += (fp - fm);
          }
        }
      }

      // y+ boundary
      constexpr int n_plus = nodes1D - 1;
      for (int i = 0; i < nodes1D; ++i) {
        const double orth = ops.nodalWeights(m, i) * metric(YH, YH, n_plus - 1, i);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += ops.nodalWeights(m, k) * ops.nodalDeriv(k, i) * metric(YH, XH, n_plus - 1, k);
        }
        for (int j = 0; j < nodes1D; ++j) {
          lhs(idx(n_plus, m), idx(j, i)) -=
              orth * ops.scsDeriv(n_plus - 1, j) + non_orth * ops.scsInterp(n_plus - 1, j);
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  void elemental_laplacian_action(
    const typename QuadViews<poly_order>::scs_tensor_array& metric,
    const typename QuadViews<poly_order>::nodal_scalar_array& scalar,
    typename QuadViews<poly_order>::nodal_scalar_array& residual
  )
  {
    /*
     * Compute the action of the LHS on a scalar field as a sequence of small (N x N), dense matrix-matrix
     * multiplications instead of a large (N^2 x N^2) matvec
     */
    constexpr unsigned nodes1D = poly_order + 1;

    // gradient at constant xhat surfaces
    ops.scs_xhat_grad(scalar, grad_phi);

    // apply metric transformation
    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order + 1; ++i) {
        integrand(j, i) = metric(XH,XH, j, i) * grad_phi(XH, j, i) + metric(XH,YH, j, i) * grad_phi(YH, j, i);
      }
    }

    // Integration of the surface fluxes
    ops.volume_1D(integrand, flux);

    // Scattering of the fluxes to nodes
    for (unsigned n = 0; n < nodes1D; ++n) {
      residual(n,0) -= flux(0,n);
      for (unsigned p = 1; p < nodes1D - 1; ++p) {
        residual(n,p) -= flux(p,n) - flux(p-1,n);
      }
      residual(n,poly_order) += flux(poly_order - 1,n);
    }

    // gradient at constant yhat surfaces
    ops.scs_yhat_grad(scalar, grad_phi);

    // apply metric transformation
    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order + 1; ++i) {
        integrand(j, i) = metric(YH,XH, j, i) * grad_phi(XH, j, i) + metric(YH,YH, j, i) * grad_phi(YH, j, i);
      }
    }

    // Integration of the surface fluxes
    ops.volume_1D(integrand, flux);

    // Scattering of the fluxes to nodes
    for (unsigned m = 0; m < nodes1D; ++m) {
      residual(0, m) -= flux(0, m);
      for (unsigned p = 1; p < nodes1D - 1; ++p) {
        residual(p, m) -= flux(p, m) - flux(p - 1, m);
      }
      residual(poly_order, m) += flux(poly_order - 1, m);
    }
  }
  //--------------------------------------------------------------------------
  void volumetric_source(
    const typename QuadViews<poly_order>::nodal_scalar_array& volume_metric,
    const typename QuadViews<poly_order>::nodal_scalar_array& nodal_source,
    typename QuadViews<poly_order>::nodal_scalar_array& rhs)
  {

    for (unsigned j = 0; j < poly_order+1; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        nodal_source(j,i) *= volume_metric(j,i);
      }
    }

    // computes the contribution of a volumetric source to the right-hand side
    ops.volume_2D(nodal_source, rhs);
  }

private:
  int idx(int i, int j) { return i*(poly_order+1)+j; };

  enum Direction {
    XH = 0,
    YH = 1
  };

  HighOrderOperatorsQuad<poly_order> ops; // has a mutable scratch

  // temp arrays
  typename QuadViews<poly_order>::nodal_vector_array grad_phi;
  typename QuadViews<poly_order>::nodal_scalar_array integrand;
  typename QuadViews<poly_order>::nodal_scalar_array flux;
};


} // namespace naluUnit
} // namespace Sierra

#endif

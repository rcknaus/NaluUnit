/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HigherOrderLaplacianQuad_h
#define HigherOrderLaplacianQuad_h

#include <element_promotion/ElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/QuadratureRule.h>
#include <NaluEnv.h>

#include <Teuchos_BLAS.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <vector>

namespace sierra {
namespace naluUnit {

template < unsigned pOrder >
class HigherOrderLaplacianQuad
{
//==========================================================================
// Class Definition
//==========================================================================
// HigherOrderLaplacianQuad - Computes elemental quantities for the assembly
// other the Poisson/Laplace equation for quadrilateral elements.
//
// TODO(rcknaus): evaluate whether templating on polynomial order is worthwhile
// TODO(rcknaus): A couple of the matrix-matrix multiplications can be merged
// TODO(rcknaus): evaluate whether padding N x (N-1) matrices to be N x N is good
// TODO(rcknaus): Profiling in general
//==========================================================================

public:
  HigherOrderLaplacianQuad(const ElementDescription& elem)
  {
    constexpr unsigned nodes1D = pOrder+1;

    // Save weighting matrix
    std::vector<double> weightVec;
    std::tie(std::ignore, weightVec) = SGL_quadrature_rule(nodes1D, elem.quadrature->scsEndLoc());
    std::copy(weightVec.begin(), weightVec.end(), nodalWeights.begin());

    // also save its transpose
    for (unsigned j = 0; j < nodes1D; ++j) {
      for (unsigned i = 0; i < nodes1D; ++i) {
        nodalWeightsT[i+j*nodes1D] = nodalWeights[j+i*nodes1D];
      }
    }

    // Mapping from a tensor-product element node ordering to the mesh's
    // element node ordering
    std::copy(elem.nodeMap.begin(), elem.nodeMap.end(), nodeMap.begin());

    // pad rectangular (N x N-1) matrices with zeros to be ( N x N )
    for (unsigned j = 0; j < nodes1D*nodes1D;++j) {
      scsInterp[j] = 0.0;
      scsDeriv[j]  = 0.0;
    }

    // Interpolation and derivative matrices evaluated at the integration points
    for (unsigned j = 0; j < pOrder; ++j) {
      for (unsigned i = 0; i < nodes1D; ++i) {
        scsInterp[i+nodes1D*j] = elem.basis->lagrange_1D(elem.scsLoc[j], i);
        scsDeriv[i+nodes1D*j] = elem.basis->lagrange_deriv_1D(elem.scsLoc[j], i);
      }
    }

    // Derivative matrix at the nodes
    // The interpolation matrix at the nodes is just the identity matrix here
    for (unsigned j = 0; j < nodes1D; ++j) {
      for (unsigned i = 0; i < nodes1D; ++i) {
        nodalDeriv[i+nodes1D*j] = elem.basis->lagrange_deriv_1D(elem.nodeLocs[i], j);
      }
    }
  }
  //--------------------------------------------------------------------------
  void diffusion_metric(
    const double* const geomR,
    const double* const geomS,
    const double* __restrict__ const coordinates,
    std::array<std::array<double, pOrder*(pOrder+1)>,4>& metric)
  {
    /*
     * Metric for the subparametric, linear mapping
     * The metric is a combination of the inverse of the Jacobian and the area-vector (A^T J^-1),
     * split into terms that are non-zero-when-orthogonal  (metricRR and metricSS)
     * and terms that are zero-when-orthogonal (metricRS and metricSR).
     *
     * TODO(rcknaus): allow full isoparametric, curved elements.
     * TODO(rcknaus): work on notation
     */
    constexpr int dim = 2;

    int deriv_offset = 0;
    for (unsigned i = 0; i < pOrder*(pOrder+1); ++i) {
      std::array<double, 8> coord_derivs;
      for (int j = 0; j < 8; ++j) {
        coord_derivs[j] = 0.0;
      }
      int coord_offset = 0;
      for (int j = 0; j < 4; ++j) {
        const double xCoord = coordinates[coord_offset+0];
        const double yCoord = coordinates[coord_offset+1];
        coord_derivs[0] += geomR[deriv_offset+0] * xCoord;
        coord_derivs[1] += geomR[deriv_offset+0] * yCoord;

        coord_derivs[2] += geomR[deriv_offset+1] * xCoord;
        coord_derivs[3] += geomR[deriv_offset+1] * yCoord;

        coord_derivs[4] += geomS[deriv_offset+0] * xCoord;
        coord_derivs[5] += geomS[deriv_offset+0] * yCoord;

        coord_derivs[6] += geomS[deriv_offset+1] * xCoord;
        coord_derivs[7] += geomS[deriv_offset+1] * yCoord;

        // update offset
        coord_offset += dim;
        deriv_offset += dim;
      }
      double inv_detj_r = 1.0 / (coord_derivs[2] * coord_derivs[1] - coord_derivs[3] * coord_derivs[0]);
      double inv_detj_s = 1.0 / (coord_derivs[6] * coord_derivs[5] - coord_derivs[7] * coord_derivs[4]);
      metric[SS][i] =  inv_detj_r * (coord_derivs[0] * coord_derivs[0] + coord_derivs[1] * coord_derivs[1]);
      metric[SR][i] = -inv_detj_r * (coord_derivs[0] * coord_derivs[2] + coord_derivs[1] * coord_derivs[3]);
      metric[RS][i] = -inv_detj_s * (coord_derivs[4] * coord_derivs[6] + coord_derivs[5] * coord_derivs[7]);
      metric[RR][i] =  inv_detj_s * (coord_derivs[6] * coord_derivs[6] + coord_derivs[7] * coord_derivs[7]);
    }
  }
  //--------------------------------------------------------------------------
  void elemental_laplacian(
    const std::array<std::array<double, pOrder * (pOrder + 1)>, 4>& metric,
    double* __restrict__ lhs)
  {
    /*
     * Computes the elemental lhs for the Laplacian operator given
     * the correct grid metrics, split into boundary and interior terms.
     *
     */
    constexpr int nodes1D = pOrder + 1;
    constexpr int nodesPerElement = (pOrder + 1) * (pOrder + 1);

    // multidimensional array views ...
    auto metric_view = [&metric, nodes1D](MetricDirection dir, int i, int j)->double {
      return metric[dir][i+nodes1D*j];
    };

    // column major
    auto lhs_view = [&, nodes1D, nodesPerElement](int m, int n, int i, int j)->double& {
      return lhs[nodeMap[i+j*nodes1D] + nodesPerElement * nodeMap[m+n*nodes1D]];
    };

    // flux past constant xi lines
    for (int n = 0; n < nodes1D; ++n) {
      // x- element boundary
      constexpr int m_minus = 0;
      for (int j = 0; j < nodes1D; ++j) {
        double orth = nodal_weights(j, n) * metric_view(RR, j, m_minus);
        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += nodal_weights(k, n) * nodal_deriv(k, j) * metric_view(RS, k, m_minus);
        }
        for (int i = 0; i < nodes1D; ++i) {
          lhs_view(m_minus, n, i, j) += orth * scs_deriv(i, m_minus) + non_orth * scs_interp(i, m_minus);
        }
      }

      // interior flux
      for (int m = 1; m < nodes1D - 1; ++m) {
        for (int j = 0; j < nodes1D; ++j) {
          const double w = nodal_weights(j, n);
          const double orthm1 = w * metric_view(RR, j, m - 1);
          const double orthp0 = w * metric_view(RR, j, m + 0);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < nodes1D; ++k) {
            const double wd = nodal_weights(k, n) * nodal_deriv(k, j);
            non_orthm1 += wd * metric_view(RS, k, m - 1);
            non_orthp0 += wd * metric_view(RS, k, m + 0);
          }

          for (int i = 0; i < nodes1D; ++i) {
            const double fm = orthm1 * scs_deriv(i, m - 1) + non_orthm1 * scs_interp(i, m - 1);
            const double fp = orthp0 * scs_deriv(i, m + 0) + non_orthp0 * scs_interp(i, m + 0);
            lhs_view(m, n, i, j) += (fp - fm);
          }
        }
      }

      // x+ element boundary
      constexpr int m_plus = nodes1D - 1;
      for (int j = 0; j < nodes1D; ++j) {
        double orth = nodal_weights(j, n) * metric_view(RR, j, m_plus - 1);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += nodal_weights(k, n) * nodal_deriv(k, j) * metric_view(RS, k, m_plus - 1);
        }
        for (int i = 0; i < nodes1D; ++i) {
          lhs_view(m_plus, n, i, j) -= orth * scs_deriv(i, m_plus - 1) + non_orth * scs_interp(i, m_plus - 1);
        }
      }
    }

    // flux past constant eta lines
    for (int m = 0; m < nodes1D; ++m) {
      // y+ boundary
      constexpr int n_minus = 0;
      for (int i = 0; i < nodes1D; ++i) {
        const double orth = nodal_weights(i, m) * metric_view(SS, i, n_minus);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += nodal_weights(k, m) * nodal_deriv(k, i) * metric_view(SR, k, n_minus);
        }
        for (int j = 0; j < nodes1D; ++j) {
          lhs_view(m, n_minus, i, j) += orth * scs_deriv(j, n_minus) + non_orth * scs_interp(j, n_minus);
        }
      }

      // interior flux past constant eta lines
      for (int n = 1; n < nodes1D - 1; ++n) {
        for (int i = 0; i < nodes1D; ++i) {
          const double orthm1 = nodal_weights(i, m) * metric_view(SS, i, n - 1);
          const double orthp0 = nodal_weights(i, m) * metric_view(SS, i, n + 0);

          double non_orthp0 = 0.0;
          double non_orthm1 = 0.0;
          for (int k = 0; k < nodes1D; ++k) {
            const double wd = nodal_weights(k, m) * nodal_deriv(k, i);
            non_orthm1 += wd * metric_view(SR, k, n - 1);
            non_orthp0 += wd * metric_view(SR, k, n + 0);
          }

          for (int j = 0; j < nodes1D; ++j) {
            const double fm = orthm1 * scs_deriv(j, n - 1) + non_orthm1 * scs_interp(j, n - 1);
            const double fp = orthp0 * scs_deriv(j, n + 0) + non_orthp0 * scs_interp(j, n + 0);
            lhs_view(m, n, i, j) += (fp - fm);
          }
        }
      }

      // y+ boundary
      constexpr int n_plus = nodes1D - 1;
      for (int i = 0; i < nodes1D; ++i) {
        const double orth = nodal_weights(i, m) * metric_view(SS, i, n_plus - 1);

        double non_orth = 0.0;
        for (int k = 0; k < nodes1D; ++k) {
          non_orth += nodal_weights(k, m) * nodal_deriv(k, i) * metric_view(SR, k, n_plus - 1);
        }
        for (int j = 0; j < nodes1D; ++j) {
          lhs_view(m, n_plus, i, j) -= orth * scs_deriv(j, n_plus - 1) + non_orth * scs_interp(j, n_plus - 1);
        }
      }
    }
  }
  //--------------------------------------------------------------------------
  void elemental_residual(
     const std::array<std::array<double, pOrder*(pOrder+1)>,4>& metric,
     const double* __restrict__ scalar,
     double* __restrict__ residual
     )
  {
    /*
     * Compute the action of the LHS on a scalar field as a sequence of small (N x N), dense matrix-matrix
     * multiplications instead of a large (N^2 x N^2) matvec
     */

    constexpr unsigned nodes1D = pOrder + 1;

    auto residual_view = [&, nodes1D] (int i, int j)->double& {
      return residual[nodeMap[i+nodes1D*j]];
    };

    // zero out scratch arrays
    for (unsigned j = 0; j < (pOrder + 1) * (pOrder + 1); ++j) {
      orth[j] = 0.0;
      temp[j] = 0.0;
      non_orth[j] = 0.0;
      flux[j] = 0.0;
    }

    // \partial_xx  term in Laplacian computation

    // orthogonal terms
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, scalar, scsDeriv.data(), orth.data());

    // non-orthogonal terms ( 2 mxm operations )
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, scalar, scsInterp.data(), temp.data());
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, nodalDeriv.data(), temp.data(), non_orth.data());

    // Hadamard with metrics
    for (unsigned j = 0; j < pOrder * (pOrder + 1); ++j) {
      temp[j] = metric[SS][j] * orth[j] + metric[SR][j] * non_orth[j];
    }

    // Integration of the surface fluxes
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, nodalWeightsT.data(), temp.data(), flux.data());

    // Scattering of the fluxes to nodes
    for (unsigned m = 0; m < nodes1D; ++m) {
      residual_view(m, 0) -= flux_view(m, 0);
      for (unsigned p = 1; p < nodes1D - 1; ++p) {
        residual_view(m, p) -= flux_view(m, p) - flux_view(m, p - 1);
      }
      residual_view(m, pOrder) += flux_view(m, pOrder - 1);
    }

    // \partial_yy term in Laplacian computation

    // zero out scratch arrays
    for (unsigned j = 0; j < (pOrder + 1) * (pOrder + 1); ++j) {
      orth[j] = 0.0;
      temp[j] = 0.0;
      non_orth[j] = 0.0;
      flux[j] = 0.0;
    }

    // orthogonal terms
    mxm(Teuchos::TRANS, Teuchos::NO_TRANS, scalar, scsDeriv.data(), orth.data());

    // non-orthogonal terms ( 2 mxm operations )
    mxm(Teuchos::TRANS, Teuchos::NO_TRANS, scalar, scsInterp.data(), temp.data());
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, nodalDeriv.data(), temp.data(), non_orth.data());

    // Hadamard with metrics
    for (unsigned j = 0; j < pOrder * (pOrder + 1); ++j) {
      temp[j] = metric[RR][j] * orth[j] + metric[RS][j] * non_orth[j];
    }

    // Integration of the surface fluxes
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, nodalWeightsT.data(), temp.data(), flux.data());

    // Scattering of the fluxes to nodes
    for (unsigned n = 0; n < nodes1D; ++n) {
      residual_view(0, n) -= flux_view(n, 0);
      for (unsigned p = 1; p < nodes1D - 1; ++p) {
        residual_view(p, n) -= flux_view(n, p) - flux_view(n, p - 1);
      }
      residual_view(pOrder, n) += flux_view(n, pOrder - 1);
    }
  }

  void volumetric_source(
    const double* source,
    double* __restrict__ rhs)
  {
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, source, nodalWeights.data(), temp.data());
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, nodalWeightsT.data(), temp.data(), vol_integral.data());

    constexpr unsigned nodesPerElement = (pOrder+1)*(pOrder+1);
    for (unsigned j = 0; j < nodesPerElement; ++j) {
      rhs[nodeMap[j]] += vol_integral[j];
    }
  }

private:

  void mxm(
    Teuchos::ETransp transA, Teuchos::ETransp transB,
    const double* A,
    const double*  B,
    double* C)
  {
   constexpr int nodes1D = pOrder+1;
   Teuchos::BLAS<int, double>().GEMM(transA, transB,
     nodes1D, nodes1D, nodes1D,
     1.0, A, nodes1D, B, nodes1D,
     0.0, C, nodes1D);
  }

  enum MetricDirection{
    RR = 0,
    RS = 1,
    SR = 2,
    SS = 3
  };

  double scs_deriv(int i, int j)  const{ return scsDeriv[i+(pOrder+1)*j]; };
  double scs_interp(int i, int j) const { return scsInterp[i+(pOrder+1)*j]; };
  double nodal_deriv(int i, int j) const{ return nodalDeriv[i+(pOrder+1)*j]; };
  double nodal_weights(int i, int j)  const{ return nodalWeights[i+(pOrder+1)*j]; };
  double flux_view(int i, int j) const { return flux[i+(pOrder+1)*j]; };

  std::array<double, (pOrder+1)*(pOrder+1)> scsDeriv;
  std::array<double, (pOrder+1)*(pOrder+1)> scsInterp;
  std::array<double, (pOrder+1)*(pOrder+1)> nodalDeriv;
  std::array<double, (pOrder+1)*(pOrder+1)> nodalWeights;
  std::array<double, (pOrder+1)*(pOrder+1)> nodalWeightsT;
  std::array<double, (pOrder+1)*(pOrder+1)> orth;
  std::array<double, (pOrder+1)*(pOrder+1)> temp;
  std::array<double, (pOrder+1)*(pOrder+1)> vol_integral;
  std::array<double, (pOrder+1)*(pOrder+1)> non_orth;
  std::array<double, (pOrder+1)*(pOrder+1)> flux;
  std::array<unsigned, (pOrder+1)*(pOrder+1)> nodeMap;
};

} // namespace naluUnit
} // namespace Sierra

#endif

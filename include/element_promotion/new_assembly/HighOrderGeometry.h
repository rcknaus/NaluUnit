/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderGeometry_h
#define HighOrderGeometry_h

#include <element_promotion/new_assembly/HighOrderOperators.h>
#include <KokkosInterface.h>

#include <tuple>
#include <vector>

namespace sierra {
namespace naluUnit {

template < unsigned poly_order>
class HighOrderGeometryQuad
{
  //==========================================================================
  // Class Definition
  //==========================================================================
  // HighOrderGeometryQuad - Computes coefficients involving mesh
  // geometric quantities for the assembly for quadrilateral elements
  //==========================================================================
public:
  //--------------------------------------------------------------------------
  HighOrderGeometryQuad(const double* nodeLocs, const double* scsLocs)
: ops(nodeLocs,scsLocs),
  jac("jacobian"),
  linear_nodal_interp("linear interpolants evaluated at node locations (times a half)"),
  linear_scs_interp("linear interpolants evaluated at scs locations (times a half)")
{
    for (unsigned j = 0; j < poly_order; ++j) {
      linear_nodal_interp(0,j) = 0.25*(1 - nodeLocs[j]);
      linear_nodal_interp(1,j) = 0.25*(1 + nodeLocs[j]);
      linear_scs_interp(0,j) = 0.25*(1 - scsLocs[j]);
      linear_scs_interp(1,j) = 0.25*(1 + scsLocs[j]);
    }
    linear_nodal_interp(0,poly_order) = 0.25*(1-nodeLocs[poly_order]);
    linear_nodal_interp(1,poly_order) = 0.25*(1+nodeLocs[poly_order]);
}
  //--------------------------------------------------------------------------
  HighOrderGeometryQuad() :
    ops(),
    jac("jacobian"),
    linear_nodal_interp("linear interpolants evaluated at node locations (times a half)"),
    linear_scs_interp("linear interpolants evaluated at scs locations (times a half)")
  {
    std::vector<double> nodeLocs; std::vector<double> scsLocs;
    std::tie(nodeLocs, std::ignore) = gauss_lobatto_legendre_rule(poly_order+1);
    std::tie(scsLocs, std::ignore)  = gauss_legendre_rule(poly_order);

    for (unsigned j = 0; j < poly_order; ++j) {
      linear_nodal_interp(0,j) = 0.25*(1 - nodeLocs[j]);
      linear_nodal_interp(1,j) = 0.25*(1 + nodeLocs[j]);
      linear_scs_interp(0,j) = 0.25*(1 - scsLocs[j]);
      linear_scs_interp(1,j) = 0.25*(1 + scsLocs[j]);
    }
    linear_nodal_interp(0,poly_order) = 0.25*(1-nodeLocs[poly_order]);
    linear_nodal_interp(1,poly_order) = 0.25*(1+nodeLocs[poly_order]);
  }
  //--------------------------------------------------------------------------
  void diffusion_metric(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::scs_tensor_array& metric)
  {
    /*
     * Metric for the full isoparametric mapping (supports curved elements)
     * The metric is a combination of the inverse of the Jacobian and the area-vector (A^T J^-1),
     */

    ops.scs_xhat_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        double inv_detj = 1.0 / (jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i));
        metric(XH,XH,j,i) =  inv_detj * (jac(XH,YH,j,i) * jac(XH,YH,j,i) + jac(YH,YH,j,i) * jac(YH,YH,j,i));
        metric(XH,YH,j,i) = -inv_detj * (jac(XH,XH,j,i) * jac(XH,YH,j,i) + jac(YH,XH,j,i) * jac(YH,YH,j,i));

      }
    }

    ops.scs_yhat_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        double inv_detj = 1.0 / (jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i));
        metric(YH,XH,j,i) = -inv_detj * (jac(XH,XH,j,i) * jac(XH,YH,j,i) + jac(YH,XH,j,i) * jac(YH,YH,j,i));
        metric(YH,YH,j,i) =  inv_detj * (jac(XH,XH,j,i) * jac(XH,XH,j,i) + jac(YH,XH,j,i) * jac(YH,XH,j,i));
      }
    }
  }
  //--------------------------------------------------------------------------
  void volume_metric(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::nodal_scalar_array& vol)
  {
    // Computes det(J) at nodes using the full isoparametric formulation

    ops.nodal_grad(coordinates, jac);

    for (unsigned j = 0; j < poly_order+1; ++j) {
      for (unsigned i = 0; i < poly_order+1; ++i) {
        vol(j,i) = jac(XH,YH, j,i) * jac(YH,XH, j,i) - jac(XH,XH,j,i) * jac(YH,YH,j,i);
      }
    }
  }
  //--------------------------------------------------------------------------
  void diffusion_metric_linear(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::scs_tensor_array& metric)
  {
    /*
     * Faster metric computation for geometrically linear elements
     */

    const double dx_x0 = coordinates(XH, poly_order, 0) - coordinates(XH, 0, 0);
    const double dx_x1 = coordinates(XH, 0, poly_order) - coordinates(XH, 0, 0);
    const double dx_y0 = coordinates(XH, poly_order, poly_order) - coordinates(XH, poly_order, 0);
    const double dx_y1 = coordinates(XH, poly_order, poly_order) - coordinates(XH, 0, poly_order);

    const double dy_x0 = coordinates(YH, poly_order, 0) - coordinates(YH, 0, 0);
    const double dy_x1 = coordinates(YH, 0, poly_order) - coordinates(YH, 0, 0);
    const double dy_y0 = coordinates(YH, poly_order, poly_order) - coordinates(YH, poly_order, 0);
    const double dy_y1 = coordinates(YH, poly_order, poly_order) - coordinates(YH, 0, poly_order);

    for (unsigned j = 0; j < poly_order; ++j) {
      const double dx_dyh = linear_scs_interp(0,j) * dx_x0 + linear_scs_interp(1,j) * dx_y1;
      const double dy_dyh = linear_scs_interp(0,j) * dy_x0 + linear_scs_interp(1,j) * dy_y1;

      const double orth = dx_dyh * dx_dyh + dy_dyh * dy_dyh;
      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dxh = linear_nodal_interp(0,i) * dx_x1 + linear_nodal_interp(1,i) * dx_y0;
        const double dy_dxh = linear_nodal_interp(0,i) * dy_x1 + linear_nodal_interp(1,i) * dy_y0;

        const double inv_detj = 1.0 / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(XH,XH,j,i) =  inv_detj * orth;
        metric(XH,YH,j,i) = -inv_detj * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
      }
    }

    for (unsigned j = 0; j < poly_order; ++j) {
      const double dx_dxh =  linear_scs_interp(0,j) * dx_x1 + linear_scs_interp(1,j) * dx_y0;
      const double dy_dxh =  linear_scs_interp(0,j) * dy_x1 + linear_scs_interp(1,j) * dy_y0;

      const double orth = dx_dxh * dx_dxh + dy_dxh * dy_dxh;
      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dyh = linear_nodal_interp(0,i) * dx_x0 + linear_nodal_interp(1,i) * dx_y1;
        const double dy_dyh = linear_nodal_interp(0,i) * dy_x0 + linear_nodal_interp(1,i) * dy_y1;

        const double inv_detj = 1.0 / (dx_dyh * dy_dxh - dx_dxh * dy_dyh);
        metric(YH,XH,j,i) = -inv_detj * (dx_dxh * dx_dyh + dy_dxh * dy_dyh);
        metric(YH,YH,j,i) =  inv_detj * orth;
      }
    }
  }
  //--------------------------------------------------------------------------
  void volume_metric_linear(
    const typename QuadViews<poly_order>::nodal_vector_array& coordinates,
    typename QuadViews<poly_order>::nodal_scalar_array& vol)
  {
    // Computes det(J) at nodes using a linear basis for element geometry

    const double dx_x0 = coordinates(XH, poly_order, 0) - coordinates(XH, 0, 0);
    const double dx_x1 = coordinates(XH, 0, poly_order) - coordinates(XH, 0, 0);
    const double dx_y0 = coordinates(XH, poly_order, poly_order) - coordinates(XH, poly_order, 0);
    const double dx_y1 = coordinates(XH, poly_order, poly_order) - coordinates(XH, 0, poly_order);

    const double dy_x0 = coordinates(YH, poly_order, 0) - coordinates(YH, 0, 0);
    const double dy_x1 = coordinates(YH, 0, poly_order) - coordinates(YH, 0, 0);
    const double dy_y0 = coordinates(YH, poly_order, poly_order) - coordinates(YH, poly_order, 0);
    const double dy_y1 = coordinates(YH, poly_order, poly_order) - coordinates(YH, 0, poly_order);

    for (unsigned j = 0; j < poly_order+1; ++j) {
      const double dx_dyh = linear_nodal_interp(0,j) * dx_x1 + linear_nodal_interp(1,j) * dx_y0;
      const double dy_dyh = linear_nodal_interp(0,j) * dy_x1 + linear_nodal_interp(1,j) * dy_y0;

      for (unsigned i = 0; i < poly_order+1; ++i) {
        const double dx_dxh = linear_nodal_interp(0,i) * dx_x0 + linear_nodal_interp(1,i) * dx_y1;
        const double dy_dxh = linear_nodal_interp(0,i) * dy_x0 + linear_nodal_interp(1,i) * dy_y1;
        vol(j,i) = dx_dyh * dy_dxh  - dx_dxh * dy_dyh;
      }
    }
  }

private:
  HighOrderOperatorsQuad<poly_order> ops; // has a mutable scratch
  typename QuadViews<poly_order>::nodal_tensor_array jac; // larger than necessary for diffusion metric eval.

  Kokkos::View<double[2][poly_order+1], array_layout> linear_nodal_interp;
  Kokkos::View<double[2][poly_order+1], array_layout> linear_scs_interp;

  enum Direction {
    XH = 0,
    YH = 1
  };

};


} // namespace naluUnit
} // namespace Sierra

#endif

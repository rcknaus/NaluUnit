/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderOperators_h
#define HighOrderOperators_h

#include <element_promotion/new_assembly/CoefficientMatrices.h>
#include <KokkosInterface.h>
#include <Teuchos_BLAS.hpp>

namespace sierra {
namespace naluUnit {

template <unsigned poly_order>
class HighOrderOperatorsQuad
{
  //==========================================================================
  // Class Definition
  //==========================================================================
  // HighOrderOperators2D - Computes commonly used derivatives/interpolations/integrations
  // for quadrilateral CVFEM elements
  //==========================================================================
public:
  // constructor with specifiable node locations / scs locations
  HighOrderOperatorsQuad(const double* nodeLocs, const double* scsLocs)
: scsDeriv(CoefficientMatrices::scs_derivative_weights<poly_order>(nodeLocs, scsLocs)),
  scsInterp(CoefficientMatrices::scs_interpolation_weights<poly_order>(nodeLocs, scsLocs)),
  nodalWeights(CoefficientMatrices::nodal_integration_weights<poly_order>(nodeLocs, scsLocs)),
  nodalDeriv(CoefficientMatrices::nodal_derivative_weights<poly_order>(nodeLocs)),
  temp("scratch") {};
  //--------------------------------------------------------------------------
  // constructor with the usual node locations / scs locations
  HighOrderOperatorsQuad()
: scsDeriv(CoefficientMatrices::scs_derivative_weights<poly_order>()),
  scsInterp(CoefficientMatrices::scs_interpolation_weights<poly_order>()),
  nodalWeights(CoefficientMatrices::nodal_integration_weights<poly_order>()),
  nodalDeriv(CoefficientMatrices::nodal_derivative_weights<poly_order>()),
  temp("scratch") {};
  //--------------------------------------------------------------------------
  void nodal_grad(
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at nodes
    Dx(&f(0,0), &grad(XH,0,0));
    Dy(&f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  void nodal_grad(
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at nodes
    Dx(&f(XH, 0,0), &grad(XH,XH,0,0));
    Dy(&f(XH, 0,0), &grad(XH,YH,0,0));
    Dx(&f(YH, 0,0), &grad(YH,XH,0,0));
    Dy(&f(YH, 0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  void scs_yhat_grad(
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element at scs of constant yhat coordinate
    Dx_yhat(&f(0,0), &grad(XH,0,0));
    Dy_yhat(&f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  void scs_yhat_grad(
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at scs of constant yhat coordinate
    Dx_yhat(&f(XH,0,0), &grad(XH,XH,0,0));
    Dy_yhat(&f(XH,0,0), &grad(XH,YH,0,0));
    Dx_yhat(&f(YH,0,0), &grad(YH,XH,0,0));
    Dy_yhat(&f(YH,0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  void scs_xhat_grad(
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_vector_array& grad)
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    Dx_xhat(&f(0,0), &grad(XH,0,0));
    Dy_xhat(&f(0,0), &grad(YH,0,0));
  }
  //--------------------------------------------------------------------------
  void scs_xhat_grad(
    const typename QuadViews<poly_order>::nodal_vector_array& f,
    typename QuadViews<poly_order>::nodal_tensor_array& grad)
  {
    // computes reference-element gradient at scs of constant xhat coordinate
    Dx_xhat(&f(XH,0,0), &grad(XH,XH,0,0));
    Dy_xhat(&f(XH,0,0), &grad(XH,YH,0,0));
    Dx_xhat(&f(YH,0,0), &grad(YH,XH,0,0));
    Dy_xhat(&f(YH,0,0), &grad(YH,YH,0,0));
  }
  //--------------------------------------------------------------------------
  void volume_1D(
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_scalar_array& f_bar)
  {
    // computes volume integral along 1D lines (e.g. "scs" in 2D)
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, nodalWeights.data(), &f(0,0), 0.0, &f_bar(0,0));
  }
  //--------------------------------------------------------------------------
  void volume_2D(
    const typename QuadViews<poly_order>::nodal_scalar_array& f,
    typename QuadViews<poly_order>::nodal_scalar_array& f_bar)
  {
    // computes volume integral along 2D volumes (e.g. "scv" in 2D)
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0,  &f(0,0), nodalWeights.data(),    0.0, temp.data());
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, nodalWeights.data(), temp.data(), 1.0, &f_bar(0,0));
  }

  const typename CoefficientMatrixViews<poly_order>::scs_matrix_array scsDeriv;
  const typename CoefficientMatrixViews<poly_order>::scs_matrix_array scsInterp;
  const typename CoefficientMatrixViews<poly_order>::nodal_matrix_array nodalWeights;
  const typename CoefficientMatrixViews<poly_order>::nodal_matrix_array nodalDeriv;

private:
  //--------------------------------------------------------------------------
  void Dx(const double* in, double* out)
  {
    // computes xhat-derivative at nodes
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, in, nodalDeriv.data(), 0.0, out);
  }
  //--------------------------------------------------------------------------
  void Dy(const double* in, double* out)
  {
    // computes yhat-derivative at nodes
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, nodalDeriv.data(), in, 0.0, out);
  }
  //--------------------------------------------------------------------------
  void Dx_xhat(const double* in, double* out)
  {
    // computes xhat-derivative at scs of constant xhat coordinate
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, in, scsDeriv.data(),  0.0, out);
  }
  //--------------------------------------------------------------------------
  void Dy_xhat(const double* in, double* out)
  {
    // computes yhat-derivative at scs of constant xhat coordinate
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, in, scsInterp.data(), 0.0, temp.data());
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, nodalDeriv.data(), temp.data(), 0.0, out);
  }
  //--------------------------------------------------------------------------
  void Dx_yhat(const double* in, double* out)
  {
    // computes xhat-derivative at scs of constant yhat coordinate
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, in, scsInterp.data(), 0.0, temp.data());
    mxm(Teuchos::TRANS   , Teuchos::NO_TRANS, 1.0, nodalDeriv.data(), temp.data(), 0.0, out);
  }
  //--------------------------------------------------------------------------
  void Dy_yhat(const double* in, double* out)
  {
    // computes yhat-derivative at scs of constant yhat coordinate
    mxm(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, in, scsDeriv.data(),  0.0, out);
  }
private:
  void mxm(
    Teuchos::ETransp transA, Teuchos::ETransp transB,
    double alpha,
    const double* A,
    const double* B,
    double beta,
    double* C)
  {
    // matrix multiplication with the usual arguments fixed
    Teuchos::BLAS<int, double>().GEMM(transA, transB,
      poly_order+1, poly_order+1, poly_order+1,
      alpha, A, poly_order+1, B, poly_order+1,
      beta, C, poly_order+1);
  }

  enum Direction
  {
    XH = 0,
    YH = 1
  };

  mutable typename QuadViews<poly_order>::nodal_scalar_array temp;
};

} // namespace naluUnit
} // namespace Sierra

#endif

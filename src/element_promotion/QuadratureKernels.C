#include <element_promotion/QuadratureKernels.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/QuadratureRule.h>

#include <Teuchos_BLAS.hpp>

#include <tuple>


namespace sierra {
namespace naluUnit {

// wrap some of the typical special quadrature operations into a class for ease-of-use
GLSQuadratureOps::GLSQuadratureOps(const ElementDescription& elem)
: blas_(Teuchos::BLAS<int,double>())
{
  std::tie(std::ignore, weightTensor_) =  GLS_quadrature_rule(elem.nodes1D, elem.quadrature->scsEndLoc());
  weightMatrix_.resize(elem.nodesPerElement*elem.nodesPerElement);
  work2D_.resize(elem.nodes1D*elem.nodes1D);

  nodes1D_ = elem.nodes1D;
  nodesPerElement_ = elem.nodesPerElement;

  p_weightTensor_ = weightTensor_.data();
  p_work2D_ = work2D_.data();

  if (elem.dimension == 3) {
    for (unsigned q = 0; q < elem.nodesPerElement; ++q) {
      const auto& lmn = elem.inverseNodeMap[q];
      for (unsigned p = 0; p < elem.nodesPerElement; ++p) {
        const auto& ijk = elem.inverseNodeMap[p];
        weightMatrix_[p + q * elem.nodesPerElement] = weightTensor_[ijk[0] + lmn[0] * elem.nodes1D]
                                                    * weightTensor_[ijk[1] + lmn[1] * elem.nodes1D]
                                                    * weightTensor_[ijk[2] + lmn[2] * elem.nodes1D];
      }
    }
    p_weightMatrix_ = weightMatrix_.data();
  }
}
//--------------------------------------------------------------------------
void
GLSQuadratureOps::volume_2D(
  const double* nodalValuesTensor,
  double* result)
{
  blas_.GEMM(
    Teuchos::NO_TRANS,
    Teuchos::NO_TRANS,
    nodes1D_, nodes1D_, nodes1D_,
    1.0,
    nodalValuesTensor, nodes1D_,
    p_weightTensor_, nodes1D_,
    0.0,
    p_work2D_, nodes1D_
  );

  blas_.GEMM(
    Teuchos::TRANS,
    Teuchos::NO_TRANS,
    nodes1D_, nodes1D_, nodes1D_,
    1.0,
    p_weightTensor_, nodes1D_,
    p_work2D_, nodes1D_,
    0.0,
    result, nodes1D_
  );
}
//--------------------------------------------------------------------------
void
GLSQuadratureOps::volume_3D(
  const double*  nodalValues,
  double* result)
{
  // TODO(rcknaus):
  // This can be computed with fewer flops by using a sequence of smaller
  // matrix-matrix multiplications.  For now, just do a big matvec.
  // This approach might be better with thread-level parallelism despite
  // the increased operation count

  blas_.GEMV(
    Teuchos::TRANS,
    nodesPerElement_, nodesPerElement_, 1.0, p_weightMatrix_,
    nodesPerElement_, nodalValues, 1, 0.0,
    result, 1
  );
}
//--------------------------------------------------------------------------
void GLSQuadratureOps::surface_2D(
  const double*  integrand,
  double* result,
  int line_offset)
{
  blas_.GEMV(
    Teuchos::TRANS,
    nodes1D_, nodes1D_,
    1.0,
    p_weightTensor_, nodes1D_,
    integrand + line_offset, 1,
    0.0,
    result + line_offset, 1
  );
}
//--------------------------------------------------------------------------
void
GLSQuadratureOps::surface_3D(
  const double* __restrict__ integrand,
  double* __restrict__ result,
  int face_offset)
{
  blas_.GEMM(
    Teuchos::NO_TRANS,
    Teuchos::NO_TRANS,
    nodes1D_, nodes1D_, nodes1D_,
    1.0,
    integrand + face_offset, nodes1D_,
    p_weightTensor_, nodes1D_,
    0.0,
    p_work2D_, nodes1D_
  );

  blas_.GEMM(
    Teuchos::TRANS,
    Teuchos::NO_TRANS,
    nodes1D_, nodes1D_, nodes1D_,
    1.0,
    p_weightTensor_, nodes1D_,
    p_work2D_, nodes1D_,
    0.0,
    result + face_offset, nodes1D_
  );
}

} // namespace naluUnit
} // namespace Sierra

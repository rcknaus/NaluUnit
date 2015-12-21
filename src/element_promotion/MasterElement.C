/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <element_promotion/MasterElement.h>
#include <ext/alloc_traits.h>
#include <cmath>
#include <memory>

namespace sierra{
namespace naluUnit{

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MasterElement::MasterElement()
  : nDim_(0),
    nodesPerElement_(0),
    numIntPoints_(0),
    scaleToStandardIsoFac_(1.0)
{
  // nothing else
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MasterElement::~MasterElement()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- isoparametric_mapping -------------------------------------------
//--------------------------------------------------------------------------
double
MasterElement::isoparametric_mapping( 
  const double b,
  const double a,
  const double xi) const
{
  return xi*(b-a)/2.0 +(a+b)/2.0;
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
HexahedralP2Element::HexahedralP2Element()
  : MasterElement(),
    scsDist_(std::sqrt(3.0)/3.0),
    useGLLGLL_(false),
    nodes1D_(3),
    numQuad_(2)
{
  // TODO(rcknaus): Avoid redundant gradient computations when useGLLGLL_ is enabled

  nDim_ = 3;
  nodesPerElement_ = nodes1D_ * nodes1D_ * nodes1D_;

  if (useGLLGLL_ && numQuad_ != 3) {
    throw std::runtime_error("useGLLGLL_ only implemented for 3-point quadrature");
  }

  // Per subcontrol-volume (surface)  quadrature rule in 1D
  // numQuad_ = 2 is the optimal quadrature for OoA
  switch (numQuad_) {
    case 1:
      gaussAbscissae_ = { 0.0 };
      gaussWeight_ = { 1.0 };
      break;
    case 2:
      gaussAbscissae_ = { -std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };
      gaussWeight_ = { 0.5, 0.5 };
      break;
    case 3:
      if (!useGLLGLL_) {
        gaussAbscissae_ = { -std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0) };
        gaussWeight_ = { 5.0/18.0, 4.0/9.0,  5.0/18.0 };
      }
      else {
        gaussAbscissae_ = { -1.0, 0.0, +1.0 };

        //use a node-specific quadrature weight with fixed integration point locations
        std::vector<std::vector<double>> weightRHS(3,std::vector<double>(3));
        weightRHS[0][0] =  (1.0-scsDist_);
        weightRHS[1][0] = -(1.0-scsDist_*scsDist_)/2.0;
        weightRHS[2][0] =  (1.0-scsDist_*scsDist_*scsDist_)/3.0;

        weightRHS[0][1] =  2.0*scsDist_;
        weightRHS[1][1] =  0.0;
        weightRHS[2][1] =  2.0*scsDist_*scsDist_*scsDist_/3.0;

        weightRHS[0][2] =  weightRHS[0][0];
        weightRHS[1][2] = -weightRHS[1][0];
        weightRHS[2][2] =  weightRHS[2][0];

        gaussWeight_.resize(numQuad_*nodes1D_);

        //left (-1) node
        gaussWeight_[0 + nodes1D_ * 0] = 0.5 * (weightRHS[2][0] - weightRHS[1][0]);
        gaussWeight_[1 + nodes1D_ * 0] = weightRHS[0][0] - weightRHS[2][0];
        gaussWeight_[2 + nodes1D_ * 0] = 0.5 * (weightRHS[2][0] + weightRHS[1][0]);

        //middle (0) node
        gaussWeight_[0 + nodes1D_ * 1] = 0.5 * (weightRHS[2][1] - weightRHS[1][1]);
        gaussWeight_[1 + nodes1D_ * 1] = weightRHS[0][1] - weightRHS[2][1];
        gaussWeight_[2 + nodes1D_ * 1] = 0.5 * (weightRHS[2][1] + weightRHS[1][1]);

        //right (+1) node
        gaussWeight_[0 + nodes1D_ * 2] = 0.5 * (weightRHS[2][2] - weightRHS[1][2]);
        gaussWeight_[1 + nodes1D_ * 2] = weightRHS[0][2] - weightRHS[2][2];
        gaussWeight_[2 + nodes1D_ * 2] = 0.5 * (weightRHS[2][2] + weightRHS[1][2]);
      }
      break;
    case 4:
      gaussAbscissae_ = {
          -std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0)),
          -std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
          +std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
          +std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0)) };

      gaussWeight_ = {
          (18.0-std::sqrt(30.0))/72.0,
          (18.0+std::sqrt(30.0))/72.0,
          (18.0+std::sqrt(30.0))/72.0,
          (18.0-std::sqrt(30.0))/72.0 };
      break;
    default:
      throw std::runtime_error("Quadrature rule not implemented");
  }

  // map the standard stk node numbering to a tensor-product style node numbering (i.e. node (m,l,k) -> m+npe*l+npe^2*k)
  stkNodeMap_ = {
                   0,  8,  1, // bottom front edge
                  11, 21,  9, // bottom mid-front edge
                   3, 10,  2, // bottom back edge
                  12, 25, 13, // mid-top front edge
                  23, 20, 24, // mid-top mid-front edge
                  15, 26, 14, // mid-top back edge
                   4, 16,  5, // top front edge
                  19, 22, 17, // top mid-front edge
                   7, 18,  6  // top back edge
                };

  // a padded list of the scs locations
  scsEndLoc_ = { -1.0, -scsDist_, scsDist_, 1.0 };
}

//--------------------------------------------------------------------------
//-------- tensor_product_node_map -----------------------------------------
//--------------------------------------------------------------------------
int
HexahedralP2Element::tensor_product_node_map(int i, int j, int k) const
{
   return stkNodeMap_[i+j*nodes1D_+k*nodes1D_*nodes1D_];
}

//--------------------------------------------------------------------------
//-------- gauss_point_location --------------------------------------------
//--------------------------------------------------------------------------
double
HexahedralP2Element::gauss_point_location(
  int nodeOrdinal,
  int gaussPointOrdinal) const
{
  double location1D;
  if (!useGLLGLL_) {
    location1D = isoparametric_mapping( scsEndLoc_[nodeOrdinal+1],
                                        scsEndLoc_[nodeOrdinal],
                                        gaussAbscissae_[gaussPointOrdinal] );
  }
  else {
    location1D = gaussAbscissae_[gaussPointOrdinal];
  }
   return location1D;
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------

double
HexahedralP2Element::tensor_product_weight(
  int s1Node, int s2Node, int s3Node,
  int s1Ip, int s2Ip, int s3Ip) const
{
  // volume integration
  double weight;
  if (!useGLLGLL_) {
    const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
    const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
    const double Ls3 = scsEndLoc_[s3Node+1]-scsEndLoc_[s3Node];
    const double isoparametricArea = Ls1 * Ls2 * Ls3;

    weight = isoparametricArea * gaussWeight_[s1Ip] * gaussWeight_[s2Ip] * gaussWeight_[s3Ip];
   }
   else {
     // weights are node-specific and take into account the isoparametric volume
     weight = gaussWeight_[s1Node+nodes1D_*s1Ip]
            * gaussWeight_[s2Node+nodes1D_*s2Ip]
            * gaussWeight_[s3Node+nodes1D_*s3Ip];
   }
   return weight;
}

//--------------------------------------------------------------------------
//-------- tensor_product_weight -------------------------------------------
//--------------------------------------------------------------------------
double
HexahedralP2Element::tensor_product_weight(
  int s1Node, int s2Node,
  int s1Ip, int s2Ip) const
{
  // surface integration
  double weight;
  if (!useGLLGLL_) {
    const double Ls1 = scsEndLoc_[s1Node+1]-scsEndLoc_[s1Node];
    const double Ls2 = scsEndLoc_[s2Node+1]-scsEndLoc_[s2Node];
    const double isoparametricArea = Ls1 * Ls2;

    weight = isoparametricArea * gaussWeight_[s1Ip] * gaussWeight_[s2Ip];
   }
   else {
     // weights are node-specific and take into account the isoparametric area
     weight = gaussWeight_[s1Node+nodes1D_*s1Ip] * gaussWeight_[s2Node+nodes1D_*s2Ip];
   }
   return weight;
}

//--------------------------------------------------------------------------
//-------- shape_fcn -------------------------------------------------------c
//--------------------------------------------------------------------------
void
HexahedralP2Element::shape_fcn(double* shpfc)
{
  for (int ip = 0; ip < numIntPoints_ * nodesPerElement_; ++ip) {
    shpfc[ip] = shapeFunctions_[ip];
  }
}

//--------------------------------------------------------------------------
//-------- eval_shape_functions_at_ips -------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_functions_at_ips()
{
  shapeFunctions_.resize(numIntPoints_*nodesPerElement_);
  hex27_shape_fcn(numIntPoints_, intgLoc_.data(), shapeFunctions_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_ips()
{
  shapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  hex27_shape_deriv(numIntPoints_, intgLoc_.data(), shapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_face_ips -----------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::eval_shape_derivs_at_face_ips()
{
  expFaceShapeDerivs_.resize(numIntPoints_*nodesPerElement_*nDim_);
  hex27_shape_deriv(numIntPoints_, intgExpFace_.data(), expFaceShapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- hex27_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::hex27_shape_fcn(
  int numIntPoints,
  const double *intgLoc,
  double *shpfc) const
{
  const double one = 1.0;
  const double half = 1.0/2.0;
  const double one4th = 1.0/4.0;
  const double one8th = 1.0/8.0;

  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    int ip_offset = nodesPerElement_*ip; // nodes per element is always 27
    int vector_offset = nDim_*ip;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];
    const double u = intgLoc[vector_offset+2];

    const double stu = s * t * u;
    const double  st  = s * t;
    const double  su  = s * u;
    const double  tu  = t * u;

    const double one_m_s = one - s;
    const double one_p_s = one + s;
    const double one_m_t = one - t;
    const double  one_p_t = one + t;
    const double  one_m_u = one - u;
    const double  one_p_u = one + u;

    const double  one_m_ss = one - s * s;
    const double one_m_tt = one - t * t;
    const double one_m_uu = one - u * u;

    shpfc[ip_offset+0]  = -one8th * stu * one_m_s  * one_m_t  * one_m_u;
    shpfc[ip_offset+1]  =  one8th * stu * one_p_s  * one_m_t  * one_m_u;
    shpfc[ip_offset+2]  = -one8th * stu * one_p_s  * one_p_t  * one_m_u;
    shpfc[ip_offset+3]  =  one8th * stu * one_m_s  * one_p_t  * one_m_u;
    shpfc[ip_offset+4]  =  one8th * stu * one_m_s  * one_m_t  * one_p_u;
    shpfc[ip_offset+5]  = -one8th * stu * one_p_s  * one_m_t  * one_p_u;
    shpfc[ip_offset+6]  =  one8th * stu * one_p_s  * one_p_t  * one_p_u;
    shpfc[ip_offset+7]  = -one8th * stu * one_m_s  * one_p_t  * one_p_u;
    shpfc[ip_offset+8]  =  one4th * tu  * one_m_ss * one_m_t  * one_m_u;
    shpfc[ip_offset+9]  = -one4th * su  * one_p_s  * one_m_tt * one_m_u;
    shpfc[ip_offset+10] = -one4th * tu  * one_m_ss * one_p_t  * one_m_u;
    shpfc[ip_offset+11] =  one4th * su  * one_m_s  * one_m_tt * one_m_u;
    shpfc[ip_offset+12] =  one4th * st  * one_m_s  * one_m_t  * one_m_uu;
    shpfc[ip_offset+13] = -one4th * st  * one_p_s  * one_m_t  * one_m_uu;
    shpfc[ip_offset+14] =  one4th * st  * one_p_s  * one_p_t  * one_m_uu;
    shpfc[ip_offset+15] = -one4th * st  * one_m_s  * one_p_t  * one_m_uu;
    shpfc[ip_offset+16] = -one4th * tu  * one_m_ss * one_m_t  * one_p_u;
    shpfc[ip_offset+17] =  one4th * su  * one_p_s  * one_m_tt * one_p_u;
    shpfc[ip_offset+18] =  one4th * tu  * one_m_ss * one_p_t  * one_p_u;
    shpfc[ip_offset+19] = -one4th * su  * one_m_s  * one_m_tt * one_p_u;
    shpfc[ip_offset+20] =                 one_m_ss * one_m_tt * one_m_uu;
    shpfc[ip_offset+21] =   -half * u   * one_m_ss * one_m_tt * one_m_u;
    shpfc[ip_offset+22] =    half * u   * one_m_ss * one_m_tt * one_p_u;
    shpfc[ip_offset+23] =   -half * s   * one_m_s  * one_m_tt * one_m_uu;
    shpfc[ip_offset+24] =    half * s   * one_p_s  * one_m_tt * one_m_uu;
    shpfc[ip_offset+25] =   -half * t   * one_m_ss * one_m_t  * one_m_uu;
    shpfc[ip_offset+26] =    half * t   * one_m_ss * one_p_t  * one_m_uu;
  }
}

//--------------------------------------------------------------------------
//-------- hex27_shape_deriv -----------------------------------------------
//--------------------------------------------------------------------------
void
HexahedralP2Element::hex27_shape_deriv(
  int numIntPoints,
  const double *intgLoc,
  double *shapeDerivs) const
{
  const double half = 1.0/2.0;
  const double one4th = 1.0/4.0;
  const double one8th = 1.0/8.0;
  const double two = 2.0;

  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    const int vector_offset = nDim_ * ip;
    const int ip_offset  = nDim_ * nodesPerElement_ * ip;
    int node; int offset;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];
    const double u = intgLoc[vector_offset+2];

    const double stu = s * t * u;
    const double st  = s * t;
    const double su  = s * u;
    const double tu  = t * u;

    const double one_m_s = 1.0 - s;
    const double one_p_s = 1.0 + s;
    const double one_m_t = 1.0 - t;
    const double one_p_t = 1.0 + t;
    const double one_m_u = 1.0 - u;
    const double one_p_u = 1.0 + u;

    const double one_m_ss = 1.0 - s * s;
    const double one_m_tt = 1.0 - t * t;
    const double one_m_uu = 1.0 - u * u;

    const double one_m_2s = 1.0 - 2.0 * s;
    const double one_m_2t = 1.0 - 2.0 * t;
    const double one_m_2u = 1.0 - 2.0 * u;

    const double one_p_2s = 1.0 + 2.0 * s;
    const double one_p_2t = 1.0 + 2.0 * t;
    const double one_p_2u = 1.0 + 2.0 * u;

    node = 0;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_m_2s * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = -one8th * su * one_m_s * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = -one8th * st * one_m_s * one_m_t * one_m_2u;

    node = 1;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_p_2s * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = one8th * su * one_p_s * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = one8th * st * one_p_s * one_m_t * one_m_2u;

    node = 2;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_p_2s * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = -one8th * su * one_p_s * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = -one8th * st * one_p_s * one_p_t * one_m_2u;

    node = 3;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_m_2s * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = one8th * su * one_m_s * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = one8th * st * one_m_s * one_p_t * one_m_2u;

    node = 4;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_m_2s * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = one8th * su * one_m_s * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = one8th * st * one_m_s * one_m_t * one_p_2u;

    node = 5;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_p_2s * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = -one8th * su * one_p_s * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = -one8th * st * one_p_s * one_m_t * one_p_2u;

    node = 6;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one8th * tu * one_p_2s * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = one8th * su * one_p_s * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = one8th * st * one_p_s * one_p_t * one_p_2u;

    node = 7;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one8th * tu * one_m_2s * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = -one8th * su * one_m_s * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = -one8th * st * one_m_s * one_p_t * one_p_2u;

    node = 8;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * stu * one_m_t * one_m_u;
    shapeDerivs[offset + 1] = one4th * u * one_m_ss * one_m_2t * one_m_u;
    shapeDerivs[offset + 2] = one4th * t * one_m_ss * one_m_t * one_m_2u;

    node = 9;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * u * one_p_2s * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = half * stu * one_p_s * one_m_u;
    shapeDerivs[offset + 2] = -one4th * s * one_p_s * one_m_tt * one_m_2u;

    node = 10;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * stu * one_p_t * one_m_u;
    shapeDerivs[offset + 1] = -one4th * u * one_m_ss * one_p_2t * one_m_u;
    shapeDerivs[offset + 2] = -one4th * t * one_m_ss * one_p_t * one_m_2u;

    node = 11;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * u * one_m_2s * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = -half * stu * one_m_s * one_m_u;
    shapeDerivs[offset + 2] = one4th * s * one_m_s * one_m_tt * one_m_2u;

    node = 12;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * t * one_m_2s * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = one4th * s * one_m_s * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = -half * stu * one_m_s * one_m_t;

    node = 13;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * t * one_p_2s * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = -one4th * s * one_p_s * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = half * stu * one_p_s * one_m_t;

    node = 14;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * t * one_p_2s * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = one4th * s * one_p_s * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = -half * stu * one_p_s * one_p_t;

    node = 15;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * t * one_m_2s * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = -one4th * s * one_m_s * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = half * stu * one_m_s * one_p_t;

    node = 16;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * stu * one_m_t * one_p_u;
    shapeDerivs[offset + 1] = -one4th * u * one_m_ss * one_m_2t * one_p_u;
    shapeDerivs[offset + 2] = -one4th * t * one_m_ss * one_m_t * one_p_2u;

    node = 17;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = one4th * u * one_p_2s * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = -half * stu * one_p_s * one_p_u;
    shapeDerivs[offset + 2] = one4th * s * one_p_s * one_m_tt * one_p_2u;

    node = 18;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * stu * one_p_t * one_p_u;
    shapeDerivs[offset + 1] = one4th * u * one_m_ss * one_p_2t * one_p_u;
    shapeDerivs[offset + 2] = one4th * t * one_m_ss * one_p_t * one_p_2u;

    node = 19;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -one4th * u * one_m_2s * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = half * stu * one_m_s * one_p_u;
    shapeDerivs[offset + 2] = -one4th * s * one_m_s * one_m_tt * one_p_2u;

    node = 20;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -two * s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = -two * t * one_m_ss * one_m_uu;
    shapeDerivs[offset + 2] = -two * u * one_m_ss * one_m_tt;

    node = 21;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = su * one_m_tt * one_m_u;
    shapeDerivs[offset + 1] = tu * one_m_ss * one_m_u;
    shapeDerivs[offset + 2] = -half * one_m_ss * one_m_tt * one_m_2u;

    node = 22;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -su * one_m_tt * one_p_u;
    shapeDerivs[offset + 1] = -tu * one_m_ss * one_p_u;
    shapeDerivs[offset + 2] = half * one_m_ss * one_m_tt * one_p_2u;

    node = 23;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -half * one_m_2s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = st * one_m_s * one_m_uu;
    shapeDerivs[offset + 2] = su * one_m_s * one_m_tt;

    node = 24;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = half * one_p_2s * one_m_tt * one_m_uu;
    shapeDerivs[offset + 1] = -st * one_p_s * one_m_uu;
    shapeDerivs[offset + 2] = -su * one_p_s * one_m_tt;

    node = 25;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = st * one_m_t * one_m_uu;
    shapeDerivs[offset + 1] = -half * one_m_ss * one_m_2t * one_m_uu;
    shapeDerivs[offset + 2] = tu * one_m_ss * one_m_t;

    node = 26;
    offset = ip_offset + nDim_ * node;
    shapeDerivs[offset + 0] = -st * one_p_t * one_m_uu;
    shapeDerivs[offset + 1] = half * one_m_ss * one_p_2t * one_m_uu;
    shapeDerivs[offset + 2] = -tu * one_m_ss * one_p_t;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Hex27SCV::Hex27SCV()
  : HexahedralP2Element()
{
  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (nodes1D_ * nodes1D_  * nodes1D_) * ( numQuad_ * numQuad_ * numQuad_); // 216

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_);
  intgLocShift_.resize(numIntPoints_*nDim_);
  ipWeight_.resize(numIntPoints_);

  // tensor product nodes (3x3x3) x tensor product quadrature (2x2x2)
  int vector_index = 0; int scalar_index = 0;
  for (int n = 0; n < nodes1D_; ++n) {
    for (int m = 0; m < nodes1D_; ++m) {
      for (int l = 0; l < nodes1D_; ++l) {

        // current node number
        const int nodeNumber = tensor_product_node_map(l,m,n);

        //tensor-product quadrature for a particular sub-cv
        for (int k = 0; k < numQuad_; ++k) {
          for (int j = 0; j < numQuad_; ++j) {
            for (int i = 0; i < numQuad_; ++i) {
              //integration point location
              intgLoc_[vector_index]     = gauss_point_location(l,i);
              intgLoc_[vector_index + 1] = gauss_point_location(m,j);
              intgLoc_[vector_index + 2] = gauss_point_location(n,k);

              //weight
              ipWeight_[scalar_index] = tensor_product_weight(l,m,n,i,j,k);

              //sub-control volume association
              ipNodeMap_[scalar_index] = nodeNumber;

              // increment indices
              ++scalar_index;
              vector_index += nDim_;
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCV::ipNodeMap(
  int /*ordinal*/)
{
  // define scv->node mappings
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
    *error = 0.0;
//  std::vector<double> jacobian(nDim_*nDim_);
  for (int k = 0; k < nelem; ++k) {
    const int scalar_elem_offset = numIntPoints_ * k;
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;

      //weighted jacobian determinant
      const double det_j = jacobian_determinant(&coords[coord_elem_offset],&shapeDerivs_[grad_offset]);

      //apply weight and store to volume
      volume[scalar_elem_offset + ip] = ipWeight_[ip] * det_j;

      //flag error
      if (det_j <= 0.0) {
        *error = 1.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- jacobian_determinant---------------------------------------------
//--------------------------------------------------------------------------
double
Hex27SCV::jacobian_determinant(
  const double *elemNodalCoords,
  const double *shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDerivs[vector_offset+0];
    const double dn_ds2 = shapeDerivs[vector_offset+1];
    const double dn_ds3 = shapeDerivs[vector_offset+2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;
  }

  const double det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
                     + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
                     + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  return det_j;
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Hex27SCS::Hex27SCS()
  : HexahedralP2Element()
{
  // set up integration rule and relevant maps on scs
  set_interior_info();

  // set up integration rule and relevant maps on faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
  eval_shape_derivs_at_face_ips();
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::set_interior_info()
{
  const int surfacesPerDirection = nodes1D_ - 1; // 2
  const int ipsPerSurface = (numQuad_*numQuad_)*(nodes1D_*nodes1D_); // 36
  const int numSurfaces = surfacesPerDirection * nDim_; // 6

  numIntPoints_ = numSurfaces*ipsPerSurface; // 216
  const int numVectorPoints = numIntPoints_*nDim_; // 648

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_); // size = 432

  // standard integration location
  intgLoc_.resize(numVectorPoints);

  // shifted
  intgLocShift_.resize(numVectorPoints);

  // Save quadrature weight and directionality information
  ipInfo_.resize(numIntPoints_);

  // a list of the scs locations in 1D
  const std::vector<double> scsLoc = { -scsDist_, scsDist_ };

  // correct orientation of area vector
  const std::vector<double> orientation = {-1.0, +1.0};

  // specify integration point locations in a dimension-by-dimension manner
  //u direction: bottom-top (0-1)
  int vector_index = 0; int lrscv_index = 0; int scalar_index = 0;
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(k,l,m);
          rightNode = tensor_product_node_map(k,l,m+1);
        }
        else {
          leftNode = tensor_product_node_map(k,l,m+1);
          rightNode = tensor_product_node_map(k,l,m);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = gauss_point_location(k,i);
            intgLoc_[vector_index + 1] = gauss_point_location(l,j);
            intgLoc_[vector_index + 2] = scsLoc[m];

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::U_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  //t direction: front-back (2-3)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(k,m,l);
          rightNode = tensor_product_node_map(k,m+1,l);
        }
        else {
          leftNode = tensor_product_node_map(k,m+1,l);
          rightNode = tensor_product_node_map(k,m,l);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = gauss_point_location(k,i);
            intgLoc_[vector_index + 1] = scsLoc[m];
            intgLoc_[vector_index + 2] = gauss_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }

  //s direction: left-right (4-5)
  for (int m = 0; m < surfacesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {
      for (int k = 0; k < nodes1D_; ++k) {

        int leftNode; int rightNode;
        if (m == 0) {
          leftNode = tensor_product_node_map(m,k,l);
          rightNode = tensor_product_node_map(m+1,k,l);
        }
        else {
          leftNode = tensor_product_node_map(m+1,k,l);
          rightNode = tensor_product_node_map(m,k,l);
        }

        for (int j = 0; j < numQuad_; ++j) {
          for (int i = 0; i < numQuad_; ++i) {
            lrscv_[lrscv_index]     = leftNode;
            lrscv_[lrscv_index + 1] = rightNode;

            intgLoc_[vector_index]     = scsLoc[m];
            intgLoc_[vector_index + 1] = gauss_point_location(k,i);
            intgLoc_[vector_index + 2] = gauss_point_location(l,j);

            //compute the quadrature weight
            ipInfo_[scalar_index].weight = -orientation[m] * tensor_product_weight(k,l,i,j);

            //direction
            ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

            ++scalar_index;
            lrscv_index += 2;
            vector_index += nDim_;
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_boundary_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::set_boundary_info()
{
  const int numFaces = 2 * nDim_; // 6
  const int nodesPerFace = nodes1D_ * nodes1D_; // 9
  ipsPerFace_ = nodesPerFace * (numQuad_ * numQuad_); // 36
  const int numFaceIps = numFaces * ipsPerFace_; // 216 = numIntPoints_ for this element

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_); // size = 648

  // face ordinal to tensor-product style node ordering
  const std::vector<int> stkFaceNodeMap = {
                                            0,  8,  1, 12, 25, 13,  4, 16,  5, // face 0(2): front face (cclockwise)
                                            1,  9,  2, 13, 24, 14,  5, 17,  6, // face 1(5): right face (cclockwise)
                                            3, 10,  2, 15, 26, 14,  6, 18,  7, // face 2(3): back face  (clockwise)
                                            0, 11,  3, 12, 23, 15,  4, 19,  7, // face 3(4): left face  (clockwise)
                                            0,  8,  1, 11, 21, 9,   3, 10,  2, // face 4(0): bottom face (clockwise)
                                            4, 16,  5, 19, 22,  17, 7, 18,  6  // face 5(1): top face (cclockwise)
                                          };


  // tensor-product style access to the map
  auto face_node_number = [=] (int i, int j, int faceOrdinal)
  {
    return stkFaceNodeMap[i + nodes1D_ * j + nodesPerFace * faceOrdinal];
  };

  // map face ip ordinal to nearest sub-control surface ip ordinal
  // sub-control surface renumbering
  const std::vector<int> faceToSurface = { 2, 5, 3, 4, 0, 1 };
  auto opp_face_map = [=] ( int k, int l, int i, int j, int face_index)
  {
    int face_offset = faceToSurface[face_index] * ipsPerFace_;

    int node_index = k + nodes1D_ * l;
    int node_offset = node_index * (numQuad_ * numQuad_);

    int ip_index = face_offset+node_offset+i+numQuad_*j;

    return ip_index;
  };

  // location of the faces in the correct order
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0, -1.0, +1.0};

  // Set points face-by-face
  int vector_index = 0; int scalar_index = 0; int faceOrdinal = 0;

  // front face: t = -1.0: counter-clockwise
  faceOrdinal = 0;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,1,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // right face: s = +1.0: counter-clockwise
  faceOrdinal = 1;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(1,k,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  // back face: s = +1.0: s-direction reversed
  faceOrdinal = 2;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = nodes1D_-1; k >= 0; --k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,1,l);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = numQuad_-1; i >= 0; --i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //left face: x = -1.0 swapped t and u
  faceOrdinal = 3;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = tensor_product_node_map(1,l,k);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index]   = oppNode;
          oppFace_[scalar_index]   = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = faceLoc[faceOrdinal];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = intgLoc_[oppFace_[scalar_index]*nDim_+2];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //bottom face: u = -1.0: swapped s and t
  faceOrdinal = 4;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(l,k,faceOrdinal);
      int oppNode = tensor_product_node_map(l,k,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(l,k,j,i,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }

  //top face: u = +1.0: counter-clockwise
  faceOrdinal = 5;
  for (int l = 0; l < nodes1D_; ++l) {
    for (int k = 0; k < nodes1D_; ++k) {
      const int nearNode = face_node_number(k,l,faceOrdinal);
      int oppNode = tensor_product_node_map(k,l,1);

      //tensor-product quadrature for a particular sub-cv
      for (int j = 0; j < numQuad_; ++j) {
        for (int i = 0; i < numQuad_; ++i) {
          // set maps
          ipNodeMap_[scalar_index] = nearNode;
          oppNode_[scalar_index] = oppNode;
          oppFace_[scalar_index] = opp_face_map(k,l,i,j,faceOrdinal);

          //integration point location
          intgExpFace_[vector_index]     = intgLoc_[oppFace_[scalar_index]*nDim_+0];
          intgExpFace_[vector_index + 1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];
          intgExpFace_[vector_index + 2] = faceLoc[faceOrdinal];

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- adjacentNodes ---------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Hex27SCS::ipNodeMap(
  int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*ipsPerFace_];
}

//--------------------------------------------------------------------------
//-------- opposingNodes --------------------------------------------------
//--------------------------------------------------------------------------
int
Hex27SCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
//-------- opposingFace --------------------------------------------------
//--------------------------------------------------------------------------
int
Hex27SCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  //returns the normal vector x_t x x_u for constant s curves
  //returns the normal vector x_u x x_s for constant t curves
  //returns the normal vector x_s x x_t for constant u curves
  std::vector<double> areaVector(nDim_);

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
        const int vector_elem_offset = nDim_ * numIntPoints_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;
      const int offset = nDim_ * ip + vector_elem_offset;

      //compute area vector for this ip
      area_vector( ipInfo_[ip].direction,
                   &coords[coord_elem_offset],
                   &shapeDerivs_[grad_offset],
                   areaVector.data() );

      // apply quadrature weight and orientation (combined as weight)
      for (int j = 0; j < nDim_; ++j) {
        areav[offset+j]  = ipInfo_[ip].weight * areaVector[j];
      }
    }
  }

  *error = 0;
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::area_vector(
  const Jacobian::Direction direction,
  const double *elemNodalCoords,
  double *shapeDeriv,
  double *areaVector) const
{

  int s1Component; int s2Component;
  switch (direction) {
    case Jacobian::S_DIRECTION:
      s1Component = static_cast<int>(Jacobian::T_DIRECTION);
      s2Component = static_cast<int>(Jacobian::U_DIRECTION);
      break;
    case Jacobian::T_DIRECTION:
      s1Component = static_cast<int>(Jacobian::S_DIRECTION);
      s2Component = static_cast<int>(Jacobian::U_DIRECTION);
      break;
    case Jacobian::U_DIRECTION:
      s1Component = static_cast<int>(Jacobian::T_DIRECTION);
      s2Component = static_cast<int>(Jacobian::S_DIRECTION);
      break;
    default:
      throw std::runtime_error("Not a valid direction for this element!");
  }

  // return the normal area vector given shape derivatives dnds OR dndt
  double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
  double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];
    const double zCoord = elemNodalCoords[vector_offset+2];

    const double dn_ds1 = shapeDeriv[vector_offset+s1Component];
    const double dn_ds2 = shapeDeriv[vector_offset+s2Component];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
  }

  //cross product
  areaVector[0] = dy_ds1*dz_ds2 - dz_ds1*dy_ds2;
  areaVector[1] = dz_ds1*dx_ds2 - dx_ds1*dz_ds2;
  areaVector[2] = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;
}

//--------------------------------------------------------------------------
//-------- grad_op ---------------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    const int scalar_elem_offset = numIntPoints_ * k;
    const int grad_elem_offset = numIntPoints_ * nDim_ * nodesPerElement_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;
      const int offset = grad_offset + grad_elem_offset;

      for (int j = 0; j < nodesPerElement_ * nDim_; ++j) {
        deriv[offset + j] = shapeDerivs_[grad_offset +j];
      }

      gradient( &coords[coord_elem_offset],
                &shapeDerivs_[grad_offset],
                &gradop[offset],
                &det_j[scalar_elem_offset+ip] );

      if (det_j[ip] <= 0.0) {
        *error = 1.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- face_grad_op ----------------------------------------------------
//--------------------------------------------------------------------------
void Hex27SCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    const int scalar_elem_offset = ipsPerFace_ * k;
    const int grad_elem_offset = ipsPerFace_ * nDim_ * nodesPerElement_ * k;

    for (int ip = 0; ip < ipsPerFace_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;
      const int offset = grad_offset + grad_elem_offset;

      gradient( &coords[coord_elem_offset],
                &expFaceShapeDerivs_[face_offset+grad_offset],
                &gradop[offset],
                &det_j[scalar_elem_offset+ip] );

      if (det_j[ip] <= 0.0) {
        *error = 1.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- gradient --------------------------------------------------------
//--------------------------------------------------------------------------
void
Hex27SCS::gradient(
  const double* elemNodalCoords,
  const double* shapeDeriv,
  double* grad,
  double* det_j) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;

  //compute Jacobian
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double zCoord = elemNodalCoords[vector_offset + 2];

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;
  }

  *det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
         + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
         + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  const double inv_det_j = (*det_j > 0.0) ? 1.0 / (*det_j) : 0.0;

  const double ds1_dx = inv_det_j*(dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3);
  const double ds2_dx = inv_det_j*(dz_ds1 * dy_ds3 - dy_ds1 * dz_ds3);
  const double ds3_dx = inv_det_j*(dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2);

  const double ds1_dy = inv_det_j*(dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3);
  const double ds2_dy = inv_det_j*(dx_ds1 * dz_ds3 - dz_ds1 * dx_ds3);
  const double ds3_dy = inv_det_j*(dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2);

  const double ds1_dz = inv_det_j*(dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3);
  const double ds2_dz = inv_det_j*(dy_ds1 * dx_ds3 - dx_ds1 * dy_ds3);
  const double ds3_dz = inv_det_j*(dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);

  // metrics
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx + dn_ds3 * ds3_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy + dn_ds3 * ds3_dy;
    grad[vector_offset + 2] = dn_ds1 * ds1_dz + dn_ds2 * ds2_dz + dn_ds3 * ds3_dz;
  }
}

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Quad93DSCS::Quad93DSCS()
  : HexahedralP2Element(),
    surfaceDimension_(2)
{
  // set up integration rule and relevant maps on scs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  eval_shape_functions_at_ips();
  eval_shape_derivs_at_ips();
}

//--------------------------------------------------------------------------
//-------- set_interior_info -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::set_interior_info()
{
  nodesPerElement_ = nodes1D_ * nodes1D_;

  std::vector<int> nodeMap = {
                               0, 4, 1,   // bottom row of nodes
                               7, 8, 5,   // middle row of nodes
                               3, 6, 2    // top row of nodes
                             };

  auto tensor_map_2D = [=] (int i, int j) { return nodeMap[i+nodes1D_*j]; };

  //1D integration rule per sub-control volume
   numIntPoints_ = (nodes1D_ * nodes1D_) * ( numQuad_ * numQuad_ ); // 36

   // define ip node mappings
   ipNodeMap_.resize(numIntPoints_);
   intgLoc_.resize(numIntPoints_*surfaceDimension_); // size = 72
   intgLocShift_.resize(numIntPoints_*surfaceDimension_); // size = 72
   ipWeight_.resize(numIntPoints_);

   // tensor product nodes (3x3) x tensor product quadrature (2x2)
   int vector_index_2D = 0; int scalar_index = 0;
   for (int l = 0; l < nodes1D_; ++l) {
     for (int k = 0; k < nodes1D_; ++k) {
       for (int j = 0; j < numQuad_; ++j) {
         for (int i = 0; i < numQuad_; ++i) {
           //integration point location
           intgLoc_[vector_index_2D]     = gauss_point_location(k,i);
           intgLoc_[vector_index_2D + 1] = gauss_point_location(l,j);

           //weight
           ipWeight_[scalar_index] = tensor_product_weight(k,l,i,j);

           //sub-control volume association
           ipNodeMap_[scalar_index] = tensor_map_2D(k,l);

           // increment indices
           ++scalar_index;
           vector_index_2D += surfaceDimension_;
         }
       }
     }
   }
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_functions_at_ips()
{
  shapeFunctions_.resize(numIntPoints_*nodesPerElement_);
  quad9_shape_fcn(numIntPoints_, intgLoc_.data(), shapeFunctions_.data());
}

//--------------------------------------------------------------------------
//-------- eval_shape_derivs_at_ips ----------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::eval_shape_derivs_at_ips()
{
  shapeDerivs_.resize(numIntPoints_*nodesPerElement_*surfaceDimension_);
  quad9_shape_deriv(numIntPoints_, intgLoc_.data(), shapeDerivs_.data());
}

//--------------------------------------------------------------------------
//-------- quad9_shape_fcn -------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::quad9_shape_fcn(
  int  numIntPoints,
  const double *intgLoc,
  double *shpfc) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    int nineIp = 9*ip; // nodes per element is always 9
    int k = 2*ip;
    const double s = intgLoc[k];
    const double t = intgLoc[k+1];

    const double one_m_s = 1.0 - s;
    const double one_p_s = 1.0 + s;
    const double one_m_t = 1.0 - t;
    const double one_p_t = 1.0 + t;

    const double one_m_ss = 1.0 - s * s;
    const double one_m_tt = 1.0 - t * t;

    shpfc[nineIp  ] =  0.25 * s * t *  one_m_s *  one_m_t;
    shpfc[nineIp+1] = -0.25 * s * t *  one_p_s *  one_m_t;
    shpfc[nineIp+2] =  0.25 * s * t *  one_p_s *  one_p_t;
    shpfc[nineIp+3] = -0.25 * s * t *  one_m_s *  one_p_t;
    shpfc[nineIp+4] = -0.50 *     t *  one_p_s *  one_m_s * one_m_t;
    shpfc[nineIp+5] =  0.50 * s     *  one_p_t *  one_m_t * one_p_s;
    shpfc[nineIp+6] =  0.50 *     t *  one_p_s *  one_m_s * one_p_t;
    shpfc[nineIp+7] = -0.50 * s     *  one_p_t *  one_m_t * one_m_s;
    shpfc[nineIp+8] =  one_m_ss * one_m_tt;
  }
}


//--------------------------------------------------------------------------
//-------- quad9_shape_deriv -----------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::quad9_shape_deriv(
  int numIntPoints,
  const double *intgLoc,
  double *deriv) const
{
  for ( int ip = 0; ip < numIntPoints; ++ip ) {
    const int grad_offset = surfaceDimension_ * nodesPerElement_ * ip; // nodes per element is always 9
    const int vector_offset = surfaceDimension_ * ip;
    int node; int offset;

    const double s = intgLoc[vector_offset+0];
    const double t = intgLoc[vector_offset+1];

    const double s2 = s*s;
    const double t2 = t*t;

    node = 0;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t - t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t - s2 + s);

    node = 1;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 - 2.0 * s * t + t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t - s2 - s);

    node = 2;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t + t2 + t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t + 2.0 * s * t + s2 + s);

    node = 3;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 0.25 * (2.0 * s * t2 + 2.0 * s * t - t2 - t);
    deriv[offset+1] = 0.25 * (2.0 * s2 * t - 2.0 * s * t + s2 - s);

    node = 4;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - s2 - 2.0 * t + 1.0);

    node = 5;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + t2 - 2.0 * s - 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + 2.0 * s * t);

    node = 6;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 + 2.0 * s * t);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t + s2 - 2.0 * t - 1.0);

    node = 7;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = -0.5 * (2.0 * s * t2 - t2 - 2.0 * s + 1.0);
    deriv[offset+1] = -0.5 * (2.0 * s2 * t - 2.0 * s * t);

    node = 8;
    offset = grad_offset + surfaceDimension_ * node;
    deriv[offset+0] = 2.0 * s * t2 - 2.0 * s;
    deriv[offset+1] = 2.0 * s2 * t - 2.0 * t;
  }
}

//--------------------------------------------------------------------------
//-------- ipNodeMap -------------------------------------------------------
//--------------------------------------------------------------------------
const int *
Quad93DSCS::ipNodeMap(
  int /*ordinal*/)
{
  // define ip->node mappings for each face (single ordinal);
  return &ipNodeMap_[0];
}

//--------------------------------------------------------------------------
//-------- determinant -----------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double * /*error*/)
{
  std::vector<double> areaVector(nDim_);

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    const int vector_elem_offset = nDim_ * numIntPoints_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = surfaceDimension_ * nodesPerElement_ * ip;
      const int offset = nDim_ * ip + vector_elem_offset;

      //compute area vector for this ip
      area_vector( &coords[coord_elem_offset],
                   &shapeDerivs_[grad_offset],
                   areaVector.data() );

      // apply quadrature weight and orientation (combined as weight)
      for (int j = 0; j < nDim_; ++j) {
        areav[offset+j]  = ipWeight_[ip] * areaVector[j];
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- area_vector -----------------------------------------------------
//--------------------------------------------------------------------------
void
Quad93DSCS::area_vector(
  const double *elemNodalCoords,
  const double *shapeDeriv,
  double *areaVector) const
{
   // return the normal area vector given shape derivatives dnds OR dndt
   double dx_ds1 = 0.0; double dy_ds1 = 0.0; double dz_ds1 = 0.0;
   double dx_ds2 = 0.0; double dy_ds2 = 0.0; double dz_ds2 = 0.0;

   for (int node = 0; node < nodesPerElement_; ++node) {
     const int vector_offset = nDim_ * node;
     const int surface_vector_offset = surfaceDimension_ * node;

     const double xCoord = elemNodalCoords[vector_offset+0];
     const double yCoord = elemNodalCoords[vector_offset+1];
     const double zCoord = elemNodalCoords[vector_offset+2];

     const double dn_ds1 = shapeDeriv[surface_vector_offset+0];
     const double dn_ds2 = shapeDeriv[surface_vector_offset+1];

     dx_ds1 += dn_ds1 * xCoord;
     dx_ds2 += dn_ds2 * xCoord;

     dy_ds1 += dn_ds1 * yCoord;
     dy_ds2 += dn_ds2 * yCoord;

     dz_ds1 += dn_ds1 * zCoord;
     dz_ds2 += dn_ds2 * zCoord;
   }

   //cross product
   areaVector[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
   areaVector[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
   areaVector[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

}  // namespace naluUnit
} // namespace sierra

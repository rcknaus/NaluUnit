/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/MasterElement.h>

#include <ext/alloc_traits.h>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

HigherOrderQuad2DSCV::HigherOrderQuad2DSCV(const ElementDescription& elem)
: MasterElement(),
  elem_(elem)
{
  nDim_ = elem_.dimension;
  nodesPerElement_ = elem_.nodesPerElement;

  // set up integration rule and relevant maps for scvs
  set_interior_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctions_ = elem_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = elem_.eval_deriv_weights(intgLoc_);
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCV::set_interior_info()
{
  //1D integration rule per sub-control volume
  numIntPoints_ = (elem_.nodes1D * elem_.nodes1D) * ( elem_.numQuad * elem_.numQuad );

  // define ip node mappings
  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_*nDim_);
  intgLocShift_.resize(numIntPoints_*nDim_);
  ipWeight_.resize(numIntPoints_);

  // tensor product nodes x tensor product quadrature
  int vector_index = 0; int scalar_index = 0;
  for (unsigned  l = 0; l < elem_.nodes1D; ++l) {
    for (unsigned  k = 0; k < elem_.nodes1D; ++k) {
      for (unsigned  j = 0; j < elem_.numQuad; ++j) {
        for (unsigned  i = 0; i < elem_.numQuad; ++i) {
          intgLoc_[vector_index]     = elem_.gauss_point_location(k,i);
          intgLoc_[vector_index + 1] = elem_.gauss_point_location(l,j);
          ipWeight_[scalar_index] = elem_.tensor_product_weight(k,l,i,j);
          ipNodeMap_[scalar_index] = elem_.tensor_product_node_map(k,l);

          // increment indices
          ++scalar_index;
          vector_index += nDim_;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCV::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctions_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctions_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCV::ipNodeMap(int /*ordinal*/)
{
 return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCV::determinant(
  const int nelem,
  const double *coords,
  double *volume,
  double *error)
{
  *error = 0.0;
  for (int k = 0; k < nelem; ++k) {
    const int scalar_elem_offset = numIntPoints_ * k;
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int grad_offset = nDim_ * nodesPerElement_ * ip;

      //weighted jacobian determinant
      const double det_j = jacobian_determinant( &coords[coord_elem_offset],
                                                 &shapeDerivs_[grad_offset] );

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
double
HigherOrderQuad2DSCV::jacobian_determinant(
  const double *elemNodalCoords,
  const double *shapeDerivs) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = node * nDim_;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];

    const double dn_ds1  = shapeDerivs[vector_offset + 0];
    const double dn_ds2  = shapeDerivs[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  const double det_j = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
  return det_j;
}
//--------------------------------------------------------------------------
HigherOrderQuad2DSCS::HigherOrderQuad2DSCS(const ElementDescription& elem)
  : MasterElement(),
    elem_(elem)
{
  nDim_ = 2;
  nodesPerElement_ = elem_.nodesPerElement;
  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctions_ = elem_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = elem_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = elem_.eval_deriv_weights(intgExpFace_);
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::set_interior_info()
{
  const int linesPerDirection = elem_.nodes1D - 1;
  const int ipsPerLine = elem_.numQuad * elem_.nodes1D;
  const int numLines = linesPerDirection * nDim_;

  numIntPoints_ = numLines * ipsPerLine;

  // define L/R mappings
  lrscv_.resize(2*numIntPoints_);

  // standard integration location
  intgLoc_.resize(numIntPoints_*nDim_);

  // shifted
  intgLocShift_.resize(numIntPoints_*nDim_);

  ipInfo_.resize(numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner

  //u-direction
  int vector_index = 0;
  int lrscv_index = 0;
  int scalar_index = 0;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (unsigned  l = 0; l < elem_.nodes1D; ++l) {

      int leftNode; int rightNode;
      if (m % 2 == 0) {
        leftNode  = elem_.tensor_product_node_map(l,m);
        rightNode = elem_.tensor_product_node_map(l,m + 1);
      }
      else {
        leftNode  = elem_.tensor_product_node_map(l,m + 1);
        rightNode = elem_.tensor_product_node_map(l,m);
      }

      for (unsigned j = 0; j < elem_.numQuad; ++j) {

        lrscv_[lrscv_index] = leftNode;
        lrscv_[lrscv_index + 1] = rightNode;

        intgLoc_[vector_index] = elem_.gauss_point_location(l,j);
        intgLoc_[vector_index + 1] = elem_.scsLoc[m];

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = std::pow(-1.0, m+1)*elem_.tensor_product_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::T_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }

  //t-direction
  for (int m = 0; m < linesPerDirection; ++m) {
    for (unsigned  l = 0; l < elem_.nodes1D; ++l) {

      int leftNode; int rightNode;
      if (m % 2 == 0) {
        leftNode  = elem_.tensor_product_node_map(m,l);
        rightNode = elem_.tensor_product_node_map(m+1,l);
      }
      else {
        leftNode  = elem_.tensor_product_node_map(m+1,l);
        rightNode = elem_.tensor_product_node_map(m,l);
      }

      for (unsigned  j = 0; j < elem_.numQuad; ++j) {

        lrscv_[lrscv_index]   = leftNode;
        lrscv_[lrscv_index+1] = rightNode;

        intgLoc_[vector_index] = elem_.scsLoc[m];
        intgLoc_[vector_index+1] = elem_.gauss_point_location(l,j);

        //compute the quadrature weight
        ipInfo_[scalar_index].weight = std::pow(-1.0, m)*elem_.tensor_product_weight(l,j);

        //direction
        ipInfo_[scalar_index].direction = Jacobian::S_DIRECTION;

        ++scalar_index;
        lrscv_index += 2;
        vector_index += nDim_;
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = elem_.nodes1D;
  ipsPerFace_ = nodesPerFace*elem_.numQuad;

  const int numFaceIps = numFaces*ipsPerFace_;

  oppFace_.resize(numFaceIps);
  ipNodeMap_.resize(numFaceIps);
  oppNode_.resize(numFaceIps);
  intgExpFace_.resize(numFaceIps*nDim_);

  auto face_node_number = [&] (int number,int faceOrdinal)
  {
    return elem_.faceNodeMap[faceOrdinal][number];
  };

  const std::vector<int> faceToLine = { 0, 3, 1, 2 };
  const std::vector<double> faceLoc = {-1.0, +1.0, +1.0, -1.0};

  int scalar_index = 0; int vector_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (unsigned k = 0; k < elem_.nodes1D; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = elem_.tensor_product_node_map(k,1);

    for (unsigned  j = 0; j < elem_.numQuad; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (unsigned  k = 0; k < elem_.nodes1D; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = elem_.tensor_product_node_map(1,k);

    for (unsigned  j = 0; j < elem_.numQuad; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }


  faceOrdinal = 2; //top face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  int elemNodeM1 = static_cast<int>(elem_.nodes1D-1);
  for (int  k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(elem_.nodes1D-k-1,faceOrdinal);
    int oppNode = elem_.tensor_product_node_map(k,1);
    for (unsigned  j = 0; j < elem_.numQuad; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index] = intgLoc_[oppFace_[scalar_index]*nDim_+0];
      intgExpFace_[vector_index+1] = faceLoc[faceOrdinal];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 3; //left face
  oppFaceIndex = 0;
  //NOTE: this faces is reversed
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(elem_.nodes1D-k-1,faceOrdinal);
    int oppNode = elem_.tensor_product_node_map(1,k);
    for (unsigned j = 0; j < elem_.numQuad; ++j) {
      ipNodeMap_[scalar_index] = nearNode;
      oppNode_[scalar_index] = oppNode;
      oppFace_[scalar_index] = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_[vector_index]   = faceLoc[faceOrdinal];
      intgExpFace_[vector_index+1] = intgLoc_[oppFace_[scalar_index]*nDim_+1];

      ++scalar_index;
      vector_index += nDim_;
      ++oppFaceIndex;
    }
  }
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctions_.size();
  for (int j =0; j < numShape; ++j) {
    shpfc[j] = shapeFunctions_[j];
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCS::ipNodeMap(int ordinal)
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_[ordinal*ipsPerFace_];
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  //returns the normal vector (dyds,-dxds) for constant t curves
  //returns the normal vector (dydt,-dxdt) for constant s curves

  std::array<double, 2> areaVector;

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;
    const int vector_elem_offset = nDim_*numIntPoints_*k;

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
void HigherOrderQuad2DSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;

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
void
HigherOrderQuad2DSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  *error = 0.0;

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
void
HigherOrderQuad2DSCS::gradient(
  const double* elemNodalCoords,
  const double* shapeDeriv,
  double* grad,
  double* det_j) const
{
  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  //compute Jacobian
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  *det_j = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;

  const double inv_det_j = (*det_j > 0.0) ? 1.0 / (*det_j) : 0.0;

  const double ds1_dx =  inv_det_j*dy_ds2;
  const double ds2_dx = -inv_det_j*dy_ds1;

  const double ds1_dy = -inv_det_j*dx_ds2;
  const double ds2_dy =  inv_det_j*dx_ds1;

  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy;
  }
}
//--------------------------------------------------------------------------
const int *
HigherOrderQuad2DSCS::adjacentNodes()
{
  // define L/R mappings
  return &lrscv_[0];
}
//--------------------------------------------------------------------------
int
HigherOrderQuad2DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
int
HigherOrderQuad2DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_[ordinal*ipsPerFace_+node];
}
//--------------------------------------------------------------------------
void
HigherOrderQuad2DSCS::area_vector(
  const Jacobian::Direction direction,
  const double *elemNodalCoords,
  double *shapeDeriv,
  double *normalVec ) const
{
  int s1Component;
  switch (direction) {
    case Jacobian::S_DIRECTION:
      s1Component = static_cast<int>(Jacobian::T_DIRECTION);
      break;
    case Jacobian::T_DIRECTION:
      s1Component = static_cast<int>(Jacobian::S_DIRECTION);
      break;
    default:
      throw std::runtime_error("Not a valid direction for this element!");
  }

  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }
  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
HigherOrderEdge2DSCS::HigherOrderEdge2DSCS(const ElementDescription& elem)
: MasterElement(),
  elem_(elem)
{
  nDim_ = 2;
  nodesPerElement_ = elem_.nodes1D;
  numIntPoints_ = elem_.numQuad * elem_.nodes1D;

  ipNodeMap_.resize(numIntPoints_);
  intgLoc_.resize(numIntPoints_);

  ipWeight_.resize(numIntPoints_);

  int scalar_index = 0;
  for (unsigned  k = 0; k < elem_.nodes1D; ++k) {
    for (unsigned  i = 0; i < elem_.numQuad; ++i) {
      intgLoc_[scalar_index]  = elem_.gauss_point_location(k,i);
      ipWeight_[scalar_index] = elem_.tensor_product_weight(k,i);
      ipNodeMap_[scalar_index] = elem_.tensor_product_node_map(k);
      ++scalar_index;
    }
  }

  shapeFunctions_ = elem_.basisBoundary->eval_basis_weights(intgLoc_);
  shapeDerivs_ = elem_.basisBoundary->eval_deriv_weights(intgLoc_);
}
//--------------------------------------------------------------------------
const int *
HigherOrderEdge2DSCS::ipNodeMap(int /*ordinal*/)
{
  return &ipNodeMap_[0];
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  std::array<double,2> areaVector;

  for (int k = 0; k < nelem; ++k) {
    const int coord_elem_offset = nDim_ * nodesPerElement_ * k;

    for (int ip = 0; ip < numIntPoints_; ++ip) {
      const int offset = nDim_ * ip + coord_elem_offset;
      const int grad_offset = ip * nodesPerElement_; // times edgeDim = 1

      // calculate the area vector
      area_vector( &coords[coord_elem_offset],
                   &shapeDerivs_[grad_offset],
                   areaVector.data() );

      // weight the area vector with the Gauss-quadrature weight for this IP
      areav[offset + 0] = ipWeight_[ip] * areaVector[0];
      areav[offset + 1] = ipWeight_[ip] * areaVector[1];
    }
  }

  // check
  *error = 0.0;
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctions_.size();
   for (int j =0; j < numShape; ++j) {
     shpfc[j] = shapeFunctions_[j];
   }
}
//--------------------------------------------------------------------------
void
HigherOrderEdge2DSCS::area_vector(
  const double* elemNodalCoords,
  const double* shapeDeriv,
  double* normalVec) const
{
  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[node] * xCoord;
    dydr += shapeDeriv[node] * yCoord;
  }
  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}

}  // namespace naluUnit
} // namespace sierra

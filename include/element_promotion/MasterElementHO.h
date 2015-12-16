/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MasterElementHO_h
#define MasterElementHO_h

#include <element_promotion/MasterElement.h>
#include <element_promotion/ElementDescription.h>
#include <vector>
#include <cstdlib>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

// 2D Quad 16 subcontrol volume
class HigherOrderQuad2DSCV : public MasterElement
{
public:
  explicit HigherOrderQuad2DSCV(const ElementDescription& elem);
  virtual ~HigherOrderQuad2DSCV() {}

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error );

  const ElementDescription& elem_;
  std::vector<double> ipWeight_;
  std::vector<double> shapeFunctions_;
  std::vector<double> shapeDerivs_;
private:
  void set_interior_info();

  double jacobian_determinant(
    const double *elemNodalCoords,
    const double *shapeDerivs ) const;
};
class HigherOrderQuad2DSCS : public MasterElement
{
public:
  explicit HigherOrderQuad2DSCS(const ElementDescription& elem);
  virtual ~HigherOrderQuad2DSCS() {}

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error );

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error );

  void gradient(
    const double* elemNodalCoords,
    const double* shapeDeriv,
    double* grad,
    double* det_j) const;

  const int * adjacentNodes();

  const int * ipNodeMap(int ordinal = 0);

  int opposingNodes(
    const int ordinal, const int node);

  int opposingFace(
    const int ordinal, const int node);

private:
  struct ContourData {
    Jacobian::Direction direction;
    double weight;
  };

  void set_interior_info();
  void set_boundary_info();

  void area_vector(
    const Jacobian::Direction direction,
    const double *elemNodalCoords,
    double *shapeDeriv,
    double *normalVec ) const;


  const ElementDescription& elem_;
  std::vector<ContourData> ipInfo_;
  int ipsPerFace_;
  std::vector<double> shapeFunctions_;
  std::vector<double> shapeDerivs_;
  std::vector<double> expFaceShapeDerivs_;
};

class HigherOrderEdge2DSCS : public MasterElement
{
public:
  explicit HigherOrderEdge2DSCS(const ElementDescription& elem);
  virtual ~HigherOrderEdge2DSCS() {}

  const int * ipNodeMap(int ordinal = 0);

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error );

  void shape_fcn(
    double *shpfc);

private:
  void area_vector(
    const double *coords,
    const double s,
    double *areaVector) const;

  const ElementDescription& elem_;
  std::vector<double> ipWeight_;
  std::vector<double> shapeFunctions_;
  std::vector<double> shapeDerivs_;
};

} // namespace nalu
} // namespace Sierra

#endif

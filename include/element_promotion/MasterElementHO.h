/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MasterElementHO_h
#define MasterElementHO_h

#include <element_promotion/MasterElement.h>
#include <vector>

namespace sierra{
namespace naluUnit{

// 2D Quad 16 subcontrol volume
struct ElementDescription;

class HigherOrderQuad2DSCV : public MasterElement
{
public:
  explicit HigherOrderQuad2DSCV(const ElementDescription& elem);
  virtual ~HigherOrderQuad2DSCV() {}

  void shape_fcn(double *shpfc) final;

  const int * ipNodeMap(int ordinal = 0) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *volume,
    double * error ) final;

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

  void shape_fcn(double *shpfc) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error) final;

  void grad_op(
    const int nelem,
    const double *coords,
    double *gradop,
    double *deriv,
    double *det_j,
    double * error) final;

  void face_grad_op(
    const int nelem,
    const int face_ordinal,
    const double *coords,
    double *gradop,
    double *det_j,
    double * error) final;

  const int * adjacentNodes() final;

  const int * ipNodeMap(int ordinal = 0) final;

  int opposingNodes(
    const int ordinal, const int node) final;

  int opposingFace(
    const int ordinal, const int node) final;

  std::vector<double> shapeFunctions_;

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

  void gradient(
    const double* elemNodalCoords,
    const double* shapeDeriv,
    double* grad,
    double* det_j) const;


  const ElementDescription& elem_;
  std::vector<ContourData> ipInfo_;
  int ipsPerFace_;

  std::vector<double> shapeDerivs_;
  std::vector<double> expFaceShapeDerivs_;
};

class HigherOrderEdge2DSCS : public MasterElement
{
public:
  explicit HigherOrderEdge2DSCS(const ElementDescription& elem);
  virtual ~HigherOrderEdge2DSCS() {}

  const int * ipNodeMap(int ordinal = 0) final;

  void determinant(
    const int nelem,
    const double *coords,
    double *areav,
    double * error ) final;

  void shape_fcn(
    double *shpfc) final;

  std::vector<double> shapeFunctions_;
private:
  void area_vector(
    const double* elemNodalCoords,
    const double* shapeDeriv,
    double* normalVec) const;

  const ElementDescription& elem_;
  std::vector<double> ipWeight_;

  std::vector<double> shapeDerivs_;
};

} // namespace nalu
} // namespace Sierra

#endif

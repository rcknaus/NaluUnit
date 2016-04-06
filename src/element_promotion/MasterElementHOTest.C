/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/MasterElementHOTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/MasterElement.h>
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/QuadratureRule.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureKernels.h>
#include <nalu_make_unique.h>
#include <TestHelper.h>

#include <Teuchos_BLAS.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

namespace sierra{
namespace naluUnit{

//==========================================================================
// MasterElementHOTests - a set of tests for higher order master elements
//
// Interpolation test: Interpolate a random polynomials of order p to random
// points within a square/cube element
//
// Derivative test: Take derivatives of random polynomials of order p at random
// points within a square/cube element
//
// Quadrature test: Integrate random polynomials of order p over the sub-cvs
// of a square/cube element
//==========================================================================
MasterElementHOTest::MasterElementHOTest(int dim, int maxOrder)
: nDim_(dim),
  polyOrder_(maxOrder),
  outputTiming_(false)
{
}
//--------------------------------------------------------------------------
void
MasterElementHOTest::execute()
{
  NaluEnv::self().naluOutputP0() << "Master Element Unit Tests for order '" << polyOrder_ << "'"<< std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;

  unsigned numTrials = 10; // number of random polynomials to interpolate/compute derivatives/integrate

  unsigned numIps = 20;    // number of randomly sampled points at which to interpolate/compute derivatives

  double tol = 1.0e-12;    // floating point tolerance (polynomial coeffs ~ order 1)
                           // Derivatives for higher polynomial orders (~10) will sometimes fail
                           // with this tolerance

  if (nDim_ == 2) {
    elem_ = ElementDescription::create(2, polyOrder_, "GaussLegendre");
    output_result("GLElement Interpolation 2D ", check_interpolation_quad(numTrials, numIps, tol));
    output_result("GLElement Derivative 2D    ", check_derivative_quad(numTrials, numIps, tol));
    output_result("GLElement Quadrature 2D    ", check_volume_quadrature_quad(numTrials,  tol));

    elem_ = ElementDescription::create(2, polyOrder_, "SGL");
    output_result("SGLElement Interpolation 2D", check_interpolation_quad(numTrials, numIps, tol));
    output_result("SGLElement Derivative 2D   ", check_derivative_quad(numTrials, numIps, tol));
    output_result("SGLElement Quadrature 2D   ", check_volume_quadrature_quad_SGL(numTrials, tol));
  }

  if (nDim_ == 3) {
    elem_ = ElementDescription::create(3, polyOrder_, "GaussLegendre");
    output_result("GLElement Interpolation 3D ", check_interpolation_hex(numTrials, numIps, tol));
    output_result("GLElement Derivative 3D    ", check_derivative_hex(numTrials, numIps, tol));
    output_result("GLElement Quadrature 3D    ", check_volume_quadrature_hex(numTrials, tol));

    elem_ = ElementDescription::create(3, polyOrder_, "SGL");
    output_result("SGLElement Interpolation 3D", check_interpolation_hex(numTrials, numIps, tol));
    output_result("SGLElement Derivative 3D   ", check_derivative_hex(numTrials, numIps, tol));
    output_result("SGLElement Quadrature 3D   ", check_volume_quadrature_hex_SGL(numTrials,  tol));
  }

  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_interpolation_quad(unsigned numTrials, unsigned numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and interpolate to a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::uniform_real_distribution<double> loc(-1.05, 1.05);

  unsigned nodesPerElement = elem_->nodesPerElement;
  std::vector<double> intgLoc(numIps * elem_->dimension);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> nodalValues(nodesPerElement);
  std::vector<double> interpWeights(numIps * nodesPerElement);
  std::vector<double> exactInterp(numIps);
  std::vector<double> approxInterp(numIps);

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned ip = 0; ip < numIps; ++ip) {
      int offset = ip * elem_->dimension;
      intgLoc[offset + 0] = loc(rng);
      intgLoc[offset + 1] = loc(rng);
    }

    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
    }

    interpWeights = elem_->eval_basis_weights(intgLoc);

    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
            nodalValues[elem_->tensor_product_node_map(i, j)] =
                  poly_val(coeffsX, elem_->nodeLocs[i])
                * poly_val(coeffsY, elem_->nodeLocs[j]);
      }
    }

    unsigned offset_1 = 0;
    for (unsigned ip = 0; ip < numIps; ++ip) {
      exactInterp[ip] =
          poly_val(coeffsX, intgLoc[offset_1 + 0])
        * poly_val(coeffsY, intgLoc[offset_1 + 1]);
      offset_1 += 2;
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      unsigned ip_offset = ip * nodesPerElement;
      double val = 0.0;
      for (unsigned nodeNumber = 0; nodeNumber < nodesPerElement; ++nodeNumber) {
        unsigned offset = ip_offset + nodeNumber;
        val += interpWeights[offset] * nodalValues[nodeNumber];
      }
      approxInterp[ip] = val;
    }

    if (is_near(approxInterp, exactInterp, tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Interpolation Test failed with max error: "
                                     << max_error(approxInterp,exactInterp) << std::endl;
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_interpolation_hex(unsigned numTrials, unsigned numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and interpolate to a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::uniform_real_distribution<double> loc(-1.05, 1.05);

  unsigned nodesPerElement = elem_->nodesPerElement;
  std::vector<double> intgLoc(numIps * elem_->dimension);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> coeffsZ(elem_->polyOrder+1);
  std::vector<double> nodalValues(nodesPerElement);
  std::vector<double> interpWeights(numIps * nodesPerElement);
  std::vector<double> exactInterp(numIps);
  std::vector<double> approxInterp(numIps);

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned ip = 0; ip < numIps; ++ip) {
      int offset = ip * elem_->dimension;
      intgLoc[offset + 0] = loc(rng);
      intgLoc[offset + 1] = loc(rng);
      intgLoc[offset + 2] = loc(rng);
    }

    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
      coeffsZ[k] = coeff(rng);
    }

    interpWeights = elem_->eval_basis_weights(intgLoc);

    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        for (unsigned k = 0; k < elem_->nodes1D; ++k) {
          nodalValues[elem_->tensor_product_node_map(i, j, k)] =
              poly_val(coeffsX, elem_->nodeLocs[i])
            * poly_val(coeffsY, elem_->nodeLocs[j])
            * poly_val(coeffsZ, elem_->nodeLocs[k]);
        }
      }
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      unsigned offset = ip*elem_->dimension;
      exactInterp[ip] =
          poly_val(coeffsX, intgLoc[offset + 0])
        * poly_val(coeffsY, intgLoc[offset + 1])
        * poly_val(coeffsZ, intgLoc[offset + 2]);
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      unsigned ip_offset = ip * nodesPerElement;
      double val = 0.0;
      for (unsigned nodeNumber = 0; nodeNumber < nodesPerElement; ++nodeNumber) {
        unsigned offset = ip_offset + nodeNumber;
        val += interpWeights[offset] * nodalValues[nodeNumber];
      }
      approxInterp[ip] = val;
    }

    if (is_near(approxInterp, exactInterp, tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Interpolation Test failed with max error: "
                                     << max_error(approxInterp,exactInterp) << std::endl;
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_derivative_quad(unsigned numTrials, unsigned numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and compute derivatives at a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1.0,1.0);
  std::uniform_real_distribution<double> loc(-1.05,1.05);

  std::vector<double> intgLoc(numIps*elem_->dimension);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> exactDeriv(numIps*elem_->dimension);
  std::vector<double> approxDeriv(numIps*elem_->dimension);

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned ip = 0; ip < numIps; ++ip) {
      int offset = ip*elem_->dimension;
      intgLoc[offset+0] = loc(rng);
      intgLoc[offset+1] = loc(rng);
    }

    std::vector<double> diffWeights = elem_->eval_deriv_weights(intgLoc);

    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
    }

    // create a (-1,1) x (-1,1) element and fill it with polynomial values
    // expect exact values to floating-point precision
    std::vector<double> nodalValues(elem_->nodesPerElement);
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for(unsigned j = 0; j < elem_->nodes1D; ++j) {
        nodalValues[elem_->tensor_product_node_map(i,j)] =
            poly_val(coeffsX,elem_->nodeLocs[i]) * poly_val(coeffsY,elem_->nodeLocs[j]);
      }
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      unsigned offset = ip*elem_->dimension;
          exactDeriv[offset + 0] =
                poly_der(coeffsX, intgLoc[offset])
              * poly_val(coeffsY, intgLoc[offset + 1]);
          exactDeriv[offset + 1] =
                poly_val(coeffsX, intgLoc[offset])
              * poly_der(coeffsY, intgLoc[offset + 1]);
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      double dndx = 0.0;
      double dndy = 0.0;
      for (unsigned node = 0; node < elem_->nodesPerElement; ++node) {
        int deriv_offset = (ip*elem_->nodesPerElement+node)*elem_->dimension;
        dndx += diffWeights[deriv_offset + 0] * nodalValues[node];
        dndy += diffWeights[deriv_offset + 1] * nodalValues[node];
      }
      approxDeriv[ip*elem_->dimension+0] = dndx;
      approxDeriv[ip*elem_->dimension+1] = dndy;
    }

    if (is_near(approxDeriv, exactDeriv, tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Derivative Test failed with max error: "
                                     << max_error(approxDeriv,exactDeriv) << std::endl;
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_derivative_hex(unsigned numTrials, unsigned numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and compute derivatives at a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1.0,1.0);
  std::uniform_real_distribution<double> loc(-1.05,1.05);

  std::vector<double> intgLoc(numIps*elem_->dimension);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> coeffsZ(elem_->polyOrder+1);
  std::vector<double> exactDeriv(numIps*elem_->dimension);
  std::vector<double> approxDeriv(numIps*elem_->dimension);

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned ip = 0; ip < numIps; ++ip) {
      int offset = ip*elem_->dimension;
      intgLoc[offset+0] = loc(rng);
      intgLoc[offset+1] = loc(rng);
      intgLoc[offset+2] = loc(rng);
    }

    std::vector<double> diffWeights = elem_->eval_deriv_weights(intgLoc);

    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
      coeffsZ[k] = coeff(rng);
    }

    std::vector<double> nodalValues(elem_->nodesPerElement);
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for(unsigned j = 0; j < elem_->nodes1D; ++j) {
        for(unsigned k = 0; k < elem_->nodes1D; ++k) {
          nodalValues[elem_->tensor_product_node_map(i,j,k)] =
              poly_val(coeffsX,elem_->nodeLocs[i])
            * poly_val(coeffsY,elem_->nodeLocs[j])
            * poly_val(coeffsZ,elem_->nodeLocs[k]);
        }
      }
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      unsigned offset = ip*elem_->dimension;
          exactDeriv[offset + 0] =
                poly_der(coeffsX, intgLoc[offset])
              * poly_val(coeffsY, intgLoc[offset + 1])
              * poly_val(coeffsZ, intgLoc[offset + 2]);
          exactDeriv[offset + 1] =
                poly_val(coeffsX, intgLoc[offset])
              * poly_der(coeffsY, intgLoc[offset + 1])
              * poly_val(coeffsZ, intgLoc[offset + 2]);
          exactDeriv[offset + 2] =
                poly_val(coeffsX, intgLoc[offset])
              * poly_val(coeffsY, intgLoc[offset + 1])
              * poly_der(coeffsZ, intgLoc[offset + 2]);
    }

    for (unsigned ip = 0; ip < numIps; ++ip) {
      double dndx = 0.0;
      double dndy = 0.0;
      double dndz = 0.0;
      for (unsigned node = 0; node < elem_->nodesPerElement; ++node) {
        int deriv_offset = (ip*elem_->nodesPerElement+node)*elem_->dimension;
        dndx += diffWeights[deriv_offset + 0] * nodalValues[node];
        dndy += diffWeights[deriv_offset + 1] * nodalValues[node];
        dndz += diffWeights[deriv_offset + 2] * nodalValues[node];
      }
      approxDeriv[ip*elem_->dimension+0] = dndx;
      approxDeriv[ip*elem_->dimension+1] = dndy;
      approxDeriv[ip*elem_->dimension+2] = dndz;
    }

    if (is_near(approxDeriv, exactDeriv, tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Derivative Test failed with max error: "
                                     << max_error(approxDeriv,exactDeriv) << std::endl;
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_volume_quadrature_quad(unsigned numTrials, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  auto masterElement = HigherOrderQuad2DSCV{*elem_};

  const auto& interpWeights  = masterElement.shapeFunctions_;
  const auto& ipWeights = masterElement.ipWeight_;
  const auto* ipNodeMap = masterElement.ipNodeMap();
  const auto& scsEndLoc = elem_->quadrature->scsEndLoc();
  std::vector<double> approxInt(elem_->nodesPerElement);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> exactInt(elem_->nodesPerElement);
  std::vector<double> nodalValues(elem_->nodesPerElement);

  double totalTime = 0.0;

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
    }

    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        nodalValues[elem_->tensor_product_node_map(i, j)] =
                      poly_val(coeffsX, elem_->nodeLocs[i])
                    * poly_val(coeffsY, elem_->nodeLocs[j]);

        exactInt[elem_->tensor_product_node_map(i, j)] =
                      poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
                    * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1]);
      }
    }

    approxInt.assign(approxInt.size(), 0.0);
    auto timeA = MPI_Wtime();
    for (int ip = 0; ip < masterElement.numIntPoints_; ++ip) {
      double interpValue = 0.0;
      for (unsigned nodeNumber = 0; nodeNumber < elem_->nodesPerElement; ++nodeNumber) {
        interpValue += interpWeights[ip*elem_->nodesPerElement+nodeNumber] * nodalValues[nodeNumber];
      }
      approxInt[ipNodeMap[ip]] += ipWeights[ip] * interpValue; //ipweights -> ws_scv_volume if not square
    }
    auto timeB = MPI_Wtime();
    totalTime += timeB-timeA;

    if (is_near(approxInt, exactInt,tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Quadrature Test failed with max error: "
                                     << max_error(approxInt,exactInt) << std::endl;
      return false;
    }
  }

  if (outputTiming_) {

    NaluEnv::self().naluOutputP0() << "Average time for volume integration loop: "
        << totalTime / numTrials << std::endl;
  }

  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_volume_quadrature_hex(unsigned numTrials, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);
  auto masterElement = HigherOrderHexSCV{*elem_};

  const auto& interpWeights  = masterElement.shapeFunctions_;
  const auto& ipWeights = masterElement.ipWeight_;
  const auto* ipNodeMap = masterElement.ipNodeMap();
  const auto& scsEndLoc = elem_->quadrature->scsEndLoc();
  std::vector<double> approxInt(elem_->nodesPerElement);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> coeffsZ(elem_->polyOrder+1);
  std::vector<double> exactInt(elem_->nodesPerElement);
  std::vector<double> nodalValues(elem_->nodesPerElement);

  double totalTime = 0.0;

  for (unsigned trial = 0; trial < numTrials; ++trial) {
    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
      coeffsZ[k] = coeff(rng);
    }

    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        for (unsigned k = 0; k < elem_->nodes1D; ++k) {
          nodalValues[elem_->tensor_product_node_map(i, j, k)] =
                poly_val(coeffsX, elem_->nodeLocs[i])
              * poly_val(coeffsY, elem_->nodeLocs[j])
              * poly_val(coeffsZ, elem_->nodeLocs[k]);

          exactInt[elem_->tensor_product_node_map(i, j, k)] =
                poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
              * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1])
              * poly_int(coeffsZ, scsEndLoc[k], scsEndLoc[k + 1]);
        }
      }
    }

    approxInt.assign(approxInt.size(), 0.0);
    auto timeA = MPI_Wtime();
    for (int ip = 0; ip < masterElement.numIntPoints_; ++ip) {
      double interpValue = 0.0;
      for (unsigned nodeNumber = 0; nodeNumber < elem_->nodesPerElement; ++nodeNumber) {
        interpValue += interpWeights[ip*elem_->nodesPerElement+nodeNumber] * nodalValues[nodeNumber];
      }
      approxInt[ipNodeMap[ip]] +=  ipWeights[ip]*interpValue;
    }
    auto timeB = MPI_Wtime();
    totalTime += timeB-timeA;

    if (is_near(approxInt, exactInt,tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Quadrature Test failed with max error: "
                                     << max_error(approxInt,exactInt) << std::endl;
      return false;
    }
  }

  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Average time for volume integration loop: "
        << totalTime / numTrials << std::endl;
  }

  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_volume_quadrature_quad_SGL(unsigned numTrials, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);
  auto masterElement = HigherOrderQuad2DSCV{*elem_};

  const auto& scsEndLoc = elem_->quadrature->scsEndLoc();
  std::vector<double> approxInt(elem_->nodesPerElement);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> exactInt(elem_->nodesPerElement);
  std::vector<double> nodalValues(elem_->nodesPerElement);
  std::vector<double> nodalValuesTensor(elem_->nodesPerElement);
  std::vector<double> approxIntTensor(elem_->nodesPerElement, 0.0);

  auto blas = Teuchos::BLAS<int,double>();
  auto quadOp = SGLQuadratureOps(*elem_);
  int nodes1D = elem_->nodes1D;

  std::vector<double> temp(elem_->nodesPerElement,0.0);

  double totalTime = 0.0;

  for (unsigned trial = 0; trial < numTrials; ++trial) {

    // get a random polyOrder-degree polynomial
    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
    }

    // exact solution
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        nodalValues[elem_->tensor_product_node_map(i, j)] =
                      poly_val(coeffsX, elem_->nodeLocs[i])
                    * poly_val(coeffsY, elem_->nodeLocs[j]);

        exactInt[elem_->tensor_product_node_map(i, j)] =
                      poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
                    * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1]);
      }
    }

    approxIntTensor.assign(nodes1D*nodes1D,0.0);


    auto timeA = MPI_Wtime();
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        // this algorithm requires the nodes to be ordered like a tensor
        nodalValuesTensor[i+nodes1D*j] =  nodalValues[elem_->tensor_product_node_map(i, j)];

        //multiply by det(J)_ij here if not a square domain
      }
    }

    quadOp.volume_2D(nodalValuesTensor.data(),approxIntTensor.data());

    // convert back to tensor-product form
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        approxInt[elem_->tensor_product_node_map(i,j)] = approxIntTensor[i+nodes1D*j];
      }
    }
    auto timeB = MPI_Wtime();

    totalTime += timeB-timeA;

    if (is_near(approxInt, exactInt,tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Quadrature Test failed with max error: "
                                     << max_error(approxInt,exactInt) << std::endl;
      return false;
    }
  }

  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Average time for volume integration loop: "
        << totalTime / numTrials << std::endl;
  }

  return testPassed;
}
//--------------------------------------------------------------------------
bool
MasterElementHOTest::check_volume_quadrature_hex_SGL(unsigned numTrials, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes
  bool testPassed = false;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);
  auto masterElement = HigherOrderHexSCV{*elem_};

  const auto& scsEndLoc = elem_->quadrature->scsEndLoc();
  std::vector<double> approxInt(elem_->nodesPerElement);
  std::vector<double> coeffsX(elem_->polyOrder+1);
  std::vector<double> coeffsY(elem_->polyOrder+1);
  std::vector<double> coeffsZ(elem_->polyOrder+1);
  std::vector<double> exactInt(elem_->nodesPerElement);
  std::vector<double> nodalValues(elem_->nodesPerElement);
  std::vector<double> nodalValuesTensor(elem_->nodesPerElement);
  std::vector<double> approxIntTensor(elem_->nodesPerElement, 0.0);

  auto blas = Teuchos::BLAS<int,double>();
  int nodes1D = elem_->nodes1D;
  int nodes2D = nodes1D*nodes1D;

  auto quadOp = SGLQuadratureOps(*elem_);
  std::vector<double> temp1(nodes2D, 0.0);
  std::vector<double> temp2(nodes2D, 0.0);

  double totalTime = 0.0;

  for (unsigned trial = 0; trial < numTrials; ++trial) {

    // get a random polyOrder-degree polynomial
    for (unsigned k = 0; k < elem_->polyOrder+1; ++k) {
      coeffsX[k] = coeff(rng);
      coeffsY[k] = coeff(rng);
      coeffsZ[k] = coeff(rng);
    }

    // exact solution
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        for (unsigned k = 0; k < elem_->nodes1D; ++k) {
          nodalValues[elem_->tensor_product_node_map(i, j, k)] =
              poly_val(coeffsX, elem_->nodeLocs[i])
            * poly_val(coeffsY, elem_->nodeLocs[j])
            * poly_val(coeffsZ, elem_->nodeLocs[k]);

          exactInt[elem_->tensor_product_node_map(i, j, k)] =
              poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
            * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1])
            * poly_int(coeffsZ, scsEndLoc[k], scsEndLoc[k + 1]);
        }
      }
    }

    approxIntTensor.assign(nodes1D * nodes1D * nodes1D, 0.0);

    auto timeA = MPI_Wtime();
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        for (unsigned k = 0; k < elem_->nodes1D; ++k) {
          // this algorithm requires the nodes to be ordered like a tensor
          nodalValuesTensor[i + nodes1D * (j + nodes1D * k)] =
              nodalValues[elem_->tensor_product_node_map(i, j, k)];

          //multiply by det(J)_ijk here if not a square domain
        }
      }
    }

    quadOp.volume_3D(nodalValues.data(), approxInt.data());

    auto timeB = MPI_Wtime();
    totalTime += timeB - timeA;

    if (is_near(approxInt, exactInt, tol)) {
      testPassed = true;
    }
    else {
      NaluEnv::self().naluOutputP0() << "Quadrature Test failed with max error: "
                                     << max_error(approxInt,exactInt) << std::endl;
      return false;
    }
  }
  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Average time for volume integration loop: "
        << totalTime / numTrials << std::endl;
  }
  return testPassed;
}
//--------------------------------------------------------------------------
double
MasterElementHOTest::poly_val(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j);
  }
  return val;
}
//--------------------------------------------------------------------------
double
MasterElementHOTest::poly_der(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j-1)*j;
  }
  return val;
}
//--------------------------------------------------------------------------
double
MasterElementHOTest::poly_int(std::vector<double> coeffs,
  double xlower, double xupper)
{
  double upper = 0.0; double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j]*std::pow(xupper,j+1)/(j+1.0);
    lower += coeffs[j]*std::pow(xlower,j+1)/(j+1.0);
  }
  return (upper-lower);
}

} // namespace naluUnit
}  // namespace sierra

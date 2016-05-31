/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/HighOrderPoissonTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/MasterElement.h>
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/QuadratureRule.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/QuadratureKernels.h>
#include <element_promotion/ElementCondenser.h>
#include <nalu_make_unique.h>
#include <Teuchos_LAPACK.hpp>
#include <TestHelper.h>

#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_io/DatabasePurpose.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/BulkDataInlinedMethods.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/environment/ReportHandler.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <limits>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// HighOrderPoissonTest - Use a four(eight)  high-order elements to solve
// the "heat conduction MMS" to effectively floating point precision
//
// 2D Test passes with 1.0e-14 error tolerance for 12 < P < 25 on my blade.
// 3D Test takes some time with high P, so it's set to P=10 and 1.0e-10
// for tolerance
//
// TODO(rcknaus):
// (maybe) a similar test using curved elements
// (maybe) change to a polynomial patch test MMS
//==========================================================================
HighOrderPoissonTest::HighOrderPoissonTest(std::string meshName)
  : meshName_(std::move(meshName)),
    order_(10),
    outputTiming_(false),
    timeCondense_(0.0),
    timeInteriorUpdate_(0.0),
    testTolerance_(1.0e-8), // 1.0e-8 is conservative even for the randomly perturbed case
    randomlyPerturbCoordinates_(true)
{
  // Nothing
}
//--------------------------------------------------------------------------
HighOrderPoissonTest::~HighOrderPoissonTest() = default;
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::execute()
{
  if (NaluEnv::self().pSize_ > 1) {
    // test is serial
    return;
  }

  double totalTime = -MPI_Wtime();

  double timeSetup = -MPI_Wtime();
  setup_mesh();
  output_banner();
  initialize_fields();
  set_output_fields();
  initialize_matrix();
  timeSetup += MPI_Wtime();

  double timeAssembly = -MPI_Wtime();
  assemble_poisson();
  timeAssembly += MPI_Wtime();

  double timeSolver = -MPI_Wtime();
  apply_dirichlet();
  solve_matrix_equation();
  timeSolver += MPI_Wtime();

  double timeUpdate = -MPI_Wtime();
  update_field();
  timeUpdate += MPI_Wtime();

  totalTime += MPI_Wtime();

  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Time to setup: "
        << timeSetup << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to assemble global matrix: "
        << timeAssembly << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to solve global matrix equation: "
        << timeSolver << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to update solution: "
        << timeUpdate << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to condense the lhs/rhs during assembly: "
        << timeCondense_ << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to compute interior update during update stage: "
        << timeInteriorUpdate_ << std::endl;

    NaluEnv::self().naluOutputP0() << "Total time for test: "
        << totalTime << std::endl;

  }
  output_results();
}
//--------------------------------------------------------------------------
struct MMSFunction {
  MMSFunction(int in_dim) : dim(in_dim), k(1.0), pi(std::acos(-1.0)) {};

  double value(const double* pos) const
  {
    double x = pos[0];
    double y = pos[1];
    double val = 0.0;
    if (dim == 2) {
      val = 0.25*(std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y));
    }
    else {
      double z = pos[2];
      val = 0.25*(std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y) + std::cos(2.0*k*pi*z));
    }
    return val;
  };

  double exact_laplacian(const double* pos) const
  {
    double x = pos[0];
    double y = pos[1];
    double source = 0.0;
    if (dim == 2) {
     source = -(k*pi)*(k*pi) * (std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y));
    }
    else {
      double z = pos[2];
      source = -(k*pi)*(k*pi) * (
          std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y) + std::cos(2.0*k*pi*z)
      );
    }
    return source;
  };

  const int dim;
  double k;
  double pi;
};
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::initialize_matrix()
{
  // count interior nodes
  const auto& node_buckets =
      bulkData_->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superPartVector_));

  // set-up connectivity
  size_t nodeNumber = 0;
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length   = b.size();
    const int* mask = stk::mesh::field_data(*mask_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      if (mask[k] == 1) {
        rowMap_.insert({b[k], nodeNumber});
        ++nodeNumber;
      }
    }
  }
  auto numNodes = rowMap_.size();
  lhs_.reshape(numNodes, numNodes);
  rhs_.resize(numNodes);
  delta_.resize(numNodes);
  lhs_.putScalar(0.0);
  rhs_.putScalar(0.0);
  delta_.putScalar(0.0);
}
//--------------------------------------------------------------------------
void HighOrderPoissonTest::apply_dirichlet()
{
  int dim = metaData_->spatial_dimension();
  int numNodes = rowMap_.size();
  auto func = MMSFunction(dim);

  const auto& face_node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK,
    stk::mesh::selectUnion(superSidePartVector_));
  for (const auto ib : face_node_buckets) {
    const auto& b = *ib;
    double* q = stk::mesh::field_data(*q_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      size_t index = rowMap_.at(b[k]);
      for (int i = 0; i < numNodes; ++i) {
        lhs_(index, i) = 0.0;
      }
      lhs_(index, index) = 1.0;
      rhs_(index) = func.value(&coords[k * dim]) - q[k];
    }
  }
}
//--------------------------------------------------------------------------
void HighOrderPoissonTest::solve_matrix_equation()
{
  Teuchos::SerialDenseSolver<int,double> solver;
  solver.setMatrix(Teuchos::rcp(&lhs_,false));
  solver.setVectors(Teuchos::rcp(&delta_,false), Teuchos::rcp(&rhs_,false));
  solver.equilibrateMatrix(); solver.equilibrateRHS();
  solver.solve();
}
//--------------------------------------------------------------------------
void HighOrderPoissonTest::assemble_poisson()
{
  int dim = elem_->dimension;
  int nodesPerElement = meSCS_->nodesPerElement_;
  int lhsSize = nodesPerElement * nodesPerElement;
  int numScsIp = meSCS_->numIntPoints_;
  int numScvIp = meSCV_->numIntPoints_;

  // allocate scratch arrays
  std::vector<double> lhs(lhsSize, 0.0);
  std::vector<double> rhs(nodesPerElement, 0.0);

  auto quadOp = SGLQuadratureOps(*elem_);
  auto condenser = ElementCondenser(*elem_);
  auto func = MMSFunction(dim);
  int numInternalNodes = condenser.num_internal_nodes();
  int numBoundaryNodes = condenser.num_boundary_nodes();
  int reducedLHSSize = numBoundaryNodes*numBoundaryNodes;
  int reducedRHSSize = numBoundaryNodes;
  std::vector<double> rlhs(reducedLHSSize, 0.0);
  std::vector<double> rrhs(reducedRHSSize, 0.0);
  std::vector<size_t> indices(reducedRHSSize,0);

  // nodal values
  std::vector<double> boundary_values(numBoundaryNodes);
  std::vector<double> interior_values(numInternalNodes);
  std::vector<double> scalarQ(nodesPerElement, 0.0);
  std::vector<double> coordinates(nodesPerElement * dim, 0.0);

  // interpolation weights
  std::vector<double> shape_functions(numScsIp * nodesPerElement, 0.0);
  meSCS_->shape_fcn(shape_functions.data());

  // derivative weights
  std::vector<double> deriv(numScsIp * nodesPerElement * dim, 0.0); // parametric coords
  std::vector<double> dndx(numScsIp * nodesPerElement * dim, 0.0); // physical coords

  // geometric information
  std::vector<double> detj_surf(numScsIp, 0.0);
  std::vector<double> areav(numScsIp*dim, 0.0);
  std::vector<double> detj_vol(numScvIp, 0.0);

  // temporary gathers specific for SGL
  std::vector<double> lhs_integrand(numScsIp * nodesPerElement, 0.0);
  std::vector<double> lhs_integrated(numScsIp * nodesPerElement, 0.0);
  std::vector<double> rhs_integrand(numScsIp, 0.0);
  std::vector<double> rhs_integrated(numScsIp, 0.0);
  std::vector<double> source_integrand(numScvIp, 0.0);
  std::vector<double> source_integrated(numScvIp, 0.0);

  // source term
  std::vector<double> nodalSource(nodesPerElement, 0.0);

  const int* lrscv = meSCS_->adjacentNodes();
  const int* ipNodeMap = meSCV_->ipNodeMap();

  const auto& buckets = bulkData_->get_buckets(stk::topology::ELEMENT_RANK,
    stk::mesh::selectUnion(superPartVector_));
  for (const auto* ib : buckets) {
    const auto& b = *ib;
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      ThrowRequire(b.num_nodes(k) == static_cast<unsigned>(nodesPerElement));
      for (int p = 0; p < lhsSize; ++p) {
        lhs[p] = 0.0;
      }

      for (int p = 0; p < nodesPerElement; ++p) {
        rhs[p] = 0.0;
      }

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        stk::mesh::Entity node = node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates_, node);
        scalarQ[ni] = *stk::mesh::field_data(*q_, node);

        // gather vectors
        const int offSet = ni*dim;
        for ( int j=0; j < dim; ++j ) {
          coordinates[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meSCS_->determinant(1, coordinates.data(), areav.data(), &scs_error);
      meSCS_->grad_op(1, coordinates.data(), dndx.data(), deriv.data(), detj_surf.data(), &scs_error);

      // Save off grad N \cdot A at ips
      // TODO(rcknaus): this can be done more cheaply using sum factorization,
      // since we're interpolating to values aligned with nodes in the orthogonal direction.
      for (int ip = 0; ip < numScsIp; ++ip) {
        double qDiff = 0.0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          double lhsfacDiff = 0.0;
          const int offSetDnDx = dim * nodesPerElement * ip + ic * dim;
          for (int j = 0; j < dim; ++j) {
            lhsfacDiff -= dndx[offSetDnDx + j] * areav[ip * dim + j];
          }
          lhs_integrand[ic * numScsIp + ip] = lhsfacDiff;
          qDiff += lhsfacDiff * scalarQ[ic];
        }
        rhs_integrand[ip] = qDiff;
      }

      // integrate surfaces
      // LHS contrib
      if (dim == 2) {
        int node_offset = 0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          quadOp.surfaces_2D(&lhs_integrand[node_offset], &lhs_integrated[node_offset]);
          node_offset += numScsIp;
        }
        quadOp.surfaces_2D(rhs_integrand.data(), rhs_integrated.data());
      }
      else {
        int node_offset = 0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          quadOp.surfaces_3D(&lhs_integrand[node_offset], &lhs_integrated[node_offset]);
          node_offset += numScsIp;
        }
        quadOp.surfaces_3D(rhs_integrand.data(), rhs_integrated.data());
      }

      // scatter
      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2 * ip];
        const int ir = lrscv[2 * ip + 1];

        const int rowL = il * nodesPerElement;
        const int rowR = ir * nodesPerElement;

        for (int ic = 0; ic < nodesPerElement; ++ic) {
          auto lhsDiff = lhs_integrated[ic * numScsIp + ip];
          lhs[rowL + ic] += lhsDiff;
          lhs[rowR + ic] -= lhsDiff;
        }
        rhs[il] -= rhs_integrated[ip];
        rhs[ir] += rhs_integrated[ip];
      }

      double scv_error = 0.0;
      meSCV_->determinant(1, coordinates.data(), detj_vol.data(), &scv_error);
      // for that time being, the 2D and 3D volume quadrature routines assume a different ordering of ips (nodes)
      if (dim == 2) {
        for (int ip = 0; ip < nodesPerElement; ++ip) {
          auto nearestNode = ipNodeMap[ip];
          nodalSource[ip] = -func.exact_laplacian(&coordinates[nearestNode * dim]) * detj_vol[ip];
        }
        quadOp.volume_2D(nodalSource.data(), source_integrated.data());

        // interpolate nodal source term to ips and assemble to nodes
        for (int ni = 0; ni < nodesPerElement; ++ni) {
          rhs[ipNodeMap[ni]] += source_integrated[ni];
        }
      }
      else {
        for (int ip = 0; ip < nodesPerElement; ++ip) {
          auto nearestNode = ipNodeMap[ip];
          nodalSource[nearestNode] = -func.exact_laplacian(&coordinates[nearestNode * dim]) * detj_vol[ip];
        }
        quadOp.volume_3D(nodalSource.data(), source_integrated.data());

        // interpolate nodal source term to ips and assemble to nodes
        for (int ni = 0; ni < nodesPerElement; ++ni) {
          rhs[ni] += source_integrated[ni];
        }
      }


      // condense out the internal degrees of freedom from the LHS/RHS
      double timeA = MPI_Wtime();
      condenser.condense(lhs.data(),rhs.data(), rlhs.data(), rrhs.data());
      double timeB = MPI_Wtime();
      timeCondense_ += timeB-timeA;

      for (int j = 0; j < numBoundaryNodes; ++j) {
        indices[j] = rowMap_.at(node_rels[j]);
      }

      for (int j = 0; j < numBoundaryNodes; ++j) {
        rhs_(indices[j]) += rrhs[j];
        for (int i = 0; i < numBoundaryNodes; ++i) {
          lhs_(indices[i], indices[j]) += rlhs[i+numBoundaryNodes*j];
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void HighOrderPoissonTest::update_field()
{
  int dim = elem_->dimension;
  int nodesPerElement = meSCS_->nodesPerElement_;
  int lhsSize = nodesPerElement * nodesPerElement;
  int numScsIp = meSCS_->numIntPoints_;
  int numScvIp = meSCV_->numIntPoints_;

  // allocate scratch arrays
  std::vector<double> lhs(lhsSize, 0.0);
  std::vector<double> rhs(nodesPerElement, 0.0);

  auto quadOp = SGLQuadratureOps(*elem_);
  auto condenser = ElementCondenser(*elem_);
  auto func = MMSFunction(dim);
  int numInternalNodes = condenser.num_internal_nodes();
  int numBoundaryNodes = condenser.num_boundary_nodes();

  // nodal values
  std::vector<double> boundary_values(numBoundaryNodes);
  std::vector<double> interior_values(numInternalNodes);
  std::vector<double> scalarQ(nodesPerElement, 0.0);
  std::vector<double> coordinates(nodesPerElement * dim, 0.0);

  // interpolation weights
  std::vector<double> shape_functions(numScsIp * nodesPerElement, 0.0);
  meSCS_->shape_fcn(shape_functions.data());

  // derivative weights
  std::vector<double> deriv(numScsIp * nodesPerElement * dim, 0.0); // parametric coords
  std::vector<double> dndx(numScsIp * nodesPerElement * dim, 0.0); // physical coords

  // geometric information
  std::vector<double> detj_surf(numScsIp, 0.0);
  std::vector<double> areav(numScsIp * dim, 0.0);
  std::vector<double> detj_vol(numScvIp, 0.0);

  // temporary gathers specific for SGL
  std::vector<double> lhs_integrand(numScsIp * nodesPerElement, 0.0);
  std::vector<double> lhs_integrated(numScsIp * nodesPerElement, 0.0);
  std::vector<double> rhs_integrand(numScsIp, 0.0);
  std::vector<double> rhs_integrated(numScsIp, 0.0);
  std::vector<double> source_integrand(numScvIp, 0.0);
  std::vector<double> source_integrated(numScvIp, 0.0);

  // source term
  std::vector<double> nodalSource(nodesPerElement, 0.0);

  const int* lrscv = meSCS_->adjacentNodes();
  const int* ipNodeMap = meSCV_->ipNodeMap();

  // update element boundaries
  auto selector =  stk::mesh::selectUnion(superPartVector_);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, selector);

  for (const auto* ib : node_buckets) {
    const auto& b = *ib;
    double* q = stk::mesh::field_data(*q_, b);
    int* mask = stk::mesh::field_data(*mask_, b);
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      if (mask[k] == 1) {
        q[k] += delta_(rowMap_.at(b[k]));
      }
    }
  }

  // update element interior
  const auto& buckets = bulkData_->get_buckets(stk::topology::ELEMENT_RANK,
    stk::mesh::selectUnion(superPartVector_));
  for (const auto* ib : buckets) {
    const auto& b = *ib;
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      ThrowRequire(b.num_nodes(k) == static_cast<unsigned>(nodesPerElement));
      for (int p = 0; p < lhsSize; ++p) {
        lhs[p] = 0.0;
      }

      for (int p = 0; p < nodesPerElement; ++p) {
        rhs[p] = 0.0;
      }

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        stk::mesh::Entity node = node_rels[ni];
        const double * coords = stk::mesh::field_data(*coordinates_, node);
        scalarQ[ni] = *stk::mesh::field_data(*q_, node);

        // gather vectors
        const int offSet = ni*dim;
        for ( int j=0; j < dim; ++j ) {
          coordinates[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meSCS_->determinant(1, coordinates.data(), areav.data(), &scs_error);
      meSCS_->grad_op(1, coordinates.data(), dndx.data(), deriv.data(), detj_surf.data(), &scs_error);

      // Save off interpolated values at surface IPs
      // TODO(rcknaus): this can be done more cheaply using sum factorization,
      //since we're interpolating to values aligned with nodes in the orthogonal direction.
      for (int ip = 0; ip < numScsIp; ++ip) {
        double qDiff = 0.0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          double lhsfacDiff = 0.0;
          const int offSetDnDx = dim * nodesPerElement * ip + ic * dim;
          for (int j = 0; j < dim; ++j) {
            lhsfacDiff -= dndx[offSetDnDx + j] * areav[ip * dim + j];
          }
          lhs_integrand[ic * numScsIp + ip] = lhsfacDiff;
          qDiff += lhsfacDiff * scalarQ[ic];
        }
        rhs_integrand[ip] = qDiff;
      }

      // integrate surfaces
      // LHS contrib
      if (dim == 2) {
        int node_offset = 0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          quadOp.surfaces_2D(&lhs_integrand[node_offset], &lhs_integrated[node_offset]);
          node_offset += numScsIp;
        }
        quadOp.surfaces_2D(rhs_integrand.data(), rhs_integrated.data());
      }
      else {
        int node_offset = 0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          quadOp.surfaces_3D(&lhs_integrand[node_offset], &lhs_integrated[node_offset]);
          node_offset += numScsIp;
        }
        quadOp.surfaces_3D(rhs_integrand.data(), rhs_integrated.data());
      }

      // scatter
      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2 * ip];
        const int ir = lrscv[2 * ip + 1];

        const int rowL = il * nodesPerElement;
        const int rowR = ir * nodesPerElement;

        for (int ic = 0; ic < nodesPerElement; ++ic) {
          auto lhsDiff = lhs_integrated[ic * numScsIp + ip];
          lhs[rowL + ic] += lhsDiff;
          lhs[rowR + ic] -= lhsDiff;
        }
        rhs[il] -= rhs_integrated[ip];
        rhs[ir] += rhs_integrated[ip];
      }

      double scv_error = 0.0;
      meSCV_->determinant(1, coordinates.data(), detj_vol.data(), &scv_error);
      if (dim == 2) {
        for (int ip = 0; ip < nodesPerElement; ++ip) {
          auto nearestNode = ipNodeMap[ip];
          nodalSource[ip] = -func.exact_laplacian(&coordinates[nearestNode * dim]) * detj_vol[ip];
        }
        quadOp.volume_2D(nodalSource.data(), source_integrated.data());

        // interpolate nodal source term to ips and assemble to nodes
        for (int ni = 0; ni < nodesPerElement; ++ni) {
          rhs[ipNodeMap[ni]] += source_integrated[ni];
        }
      }
      else {
        for (int ip = 0; ip < nodesPerElement; ++ip) {
          auto nearestNode = ipNodeMap[ip];
          nodalSource[nearestNode] = -func.exact_laplacian(&coordinates[nearestNode * dim]) * detj_vol[ip];
        }
        quadOp.volume_3D(nodalSource.data(), source_integrated.data());

        // interpolate nodal source term to ips and assemble to nodes
        for (int ni = 0; ni < nodesPerElement; ++ni) {
          rhs[ni] += source_integrated[ni];
        }
      }

      for (int j = 0; j < numBoundaryNodes; ++j) {
        boundary_values[j] = scalarQ[j];
      }

      double timeA = MPI_Wtime();
      condenser.compute_interior_update(
        lhs.data(), rhs.data(),
        boundary_values.data(), interior_values.data()
      );
      double timeB = MPI_Wtime();
      timeInteriorUpdate_ += timeB-timeA;

      for (int j = numBoundaryNodes; j < nodesPerElement; ++j) {
        *static_cast<double*>(stk::mesh::field_data(*q_, node_rels[j])) += interior_values[j - numBoundaryNodes];
      }
    }
  }
}
//--------------------------------------------------------------------------
bool HighOrderPoissonTest::check_solution()
{

  double maxError = -1.0;
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK,
    stk::mesh::selectUnion(superPartVector_));
  for (const auto ib : node_buckets) {
    const auto& b = *ib;
    const auto length = b.size();
    double* q = stk::mesh::field_data(*q_, b);
    double* qExact = stk::mesh::field_data(*qExact_, b);
    for (size_t k = 0; k < length; ++k) {
      if (std::isnan(q[k])) {
        NaluEnv::self().naluOutputP0()
            << "Poisson test experienced a nan at GID, " << bulkData_->identifier(b[k]) << " ";
        return false;
      }
      maxError = std::max(maxError, std::abs(q[k] - qExact[k]));
    }
  }

  // error should be small for high order
  if (maxError >= testTolerance_) {
    NaluEnv::self().naluOutputP0()
        << "Poisson test failed with a maximum error of "
        << maxError << " vs a tolerance of "
        << testTolerance_ << ", ";
  }
  return (maxError < testTolerance_);
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::setup_mesh()
{
  stk::ParallelMachine pm = NaluEnv::self().parallel_comm();

  //mesh setup
  metaData_ = make_unique<stk::mesh::MetaData>();
  bulkData_ = make_unique<stk::mesh::BulkData>(*metaData_, pm, stk::mesh::BulkData::NO_AUTO_AURA);
  ioBroker_ = make_unique<stk::io::StkMeshIoBroker>(pm);
  ioBroker_->set_bulk_data(*bulkData_);

  // deal with input mesh
  ioBroker_->add_mesh_database(meshName_, stk::io::READ_MESH);
  ioBroker_->create_input_mesh();

  elem_ = ElementDescription::create(metaData_->spatial_dimension(), order_, "SGL", true);
  ThrowRequire(elem_.get() != nullptr);
  meSCV_ = make_master_volume_element(*elem_);
  meSCS_ = make_master_subcontrol_surface_element(*elem_);

  setup_super_parts();
  register_fields();

  // populate bulk data
  ioBroker_->populate_bulk_data();

  if (randomlyPerturbCoordinates_) {
    perturb_coordinates(0.5, 0.15);
  }

  bulkData_->modification_begin();
  PromoteElement(*elem_).promote_elements(
    originalPartVector_,
    *coordinates_,
    *bulkData_
  );
  bulkData_->modification_end();
}
//--------------------------------------------------------------------------
void HighOrderPoissonTest::output_banner()
{
  std::string elemType;
  if(metaData_->spatial_dimension() == 2) {
    unsigned nodes = (order_+1)*(order_+1);
    elemType = "Quad" + std::to_string(nodes);
  }
  else {
    unsigned nodes = (order_+1)*(order_+1)*(order_+1);
    elemType = "Hex" + std::to_string(nodes);
  }
  fineOutputName_   = "test_output/" + elemType + ".e";

  NaluEnv::self().naluOutputP0()
      << "Using '" << elemType
      << "' Elements with quadrature type '" << "SGL" << "' to solve a Poisson equation MMS"
      <<   std::endl;

  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
HighOrderPoissonTest::make_master_volume_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCV>(elem);
  }
  return make_unique<HigherOrderHexSCV>(elem);
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
HighOrderPoissonTest::make_master_subcontrol_surface_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCS>(elem);
  }
  return make_unique<HigherOrderHexSCS>(elem);
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::register_fields()
{
  for (auto* basePart : originalPartVector_) {
    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_, *basePart, metaData_->spatial_dimension());

    mask_ = &(metaData_-> declare_field<ScalarIntFieldType>(stk::topology::NODE_RANK, "boundary_mask"));
    stk::mesh::put_field(*mask_, *basePart);

    q_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *basePart);

    qExact_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "exact_scalar"));
    stk::mesh::put_field(*qExact_, *basePart);
  }

  // save space for parts of the input mesh
  for (auto* superPart : superPartVector_) {
    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_, *superPart, metaData_->spatial_dimension());

    mask_ = &(metaData_-> declare_field<ScalarIntFieldType>(stk::topology::NODE_RANK, "boundary_mask"));
    stk::mesh::put_field(*mask_, *superPart);

    q_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *superPart);

    qExact_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "exact_scalar"));
    stk::mesh::put_field(*qExact_, *superPart);
  }
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::setup_super_parts()
{
  originalPartVector_ = metaData_->get_mesh_parts();
  for (auto* targetPart : originalPartVector_) {
    if (targetPart->topology().rank() == stk::topology::ELEM_RANK) {
      auto* superElemPart = &metaData_->declare_part_with_topology(
        super_element_part_name(targetPart->name()),
        stk::create_superelement_topology(static_cast<unsigned>(elem_->nodesPerElement))
      );

      stk::io::put_io_part_attribute(*superElemPart);
      superPartVector_.push_back(superElemPart);
    }
    else if (!targetPart->subsets().empty()) {
      auto* superSuperset = &metaData_->declare_part(super_element_part_name(targetPart->name()));
      for (const auto* subset : targetPart->subsets()) {
        if (subset->topology().rank() == metaData_->side_rank()) {
          stk::mesh::Part* superFacePart;
          if (metaData_->spatial_dimension() == 2) {
            superFacePart = &metaData_->declare_part_with_topology(
              super_subset_part_name(subset->name(), elem_->nodesPerElement, elem_->nodesPerFace),
              stk::create_superedge_topology(static_cast<unsigned>(elem_->nodesPerFace))
            );
          }
          else {
            superFacePart = &metaData_->declare_part_with_topology(
              super_subset_part_name(subset->name(), elem_->nodesPerElement, elem_->nodesPerFace),
              stk::create_superface_topology(static_cast<unsigned>(elem_->nodesPerFace))
            );
          }
          superSidePartVector_.push_back(superFacePart);
          superPartVector_.push_back(superFacePart);
          metaData_->declare_part_subset(*superSuperset, *superFacePart);
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::set_output_fields()
{
  promoteIO_ = make_unique<PromotedElementIO>(
    *elem_,
    *metaData_,
    *bulkData_,
    originalPartVector_,
    fineOutputName_
  );
  promoteIO_->add_fields({q_, qExact_});
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::perturb_coordinates(double elem_size, double fac)
{
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-fac*elem_size, fac*elem_size);

  auto selector = stk::mesh::selectUnion(originalPartVector_);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, selector);

  int dim = metaData_->spatial_dimension();
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length  = b.size();
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      for (int j = 0; j < dim; ++j) {
        coords[k*dim+j] += coeff(rng);
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::initialize_fields()
{
  int dim = metaData_->spatial_dimension();
  auto func = MMSFunction(dim);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superPartVector_));
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length  = b.size();
    double* q = stk::mesh::field_data(*q_, b);
    int* mask = stk::mesh::field_data(*mask_, b);
    double* qExact = stk::mesh::field_data(*qExact_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      mask[k] = 0;
      q[k] = 0.0;
      qExact[k] = func.value(&coords[k*dim]);
    }
  }

  const auto& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superPartVector_));

  const auto numBoundaryNodes = ElementCondenser(*elem_).num_boundary_nodes();
  for (const auto ib : elem_buckets ) {
    const auto& b = *ib ;
    const auto length   = b.size();
    for ( size_t k = 0 ; k < length ; ++k ) {
      const auto* node_rels = b.begin_nodes(k);
      for (int j = 0; j < numBoundaryNodes; ++j) {
        *static_cast<int*>(stk::mesh::field_data(*mask_, node_rels[j])) = 1;
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HighOrderPoissonTest::output_results()
{
  output_result("Poisson", check_solution());
  promoteIO_->write_database_data(0.0);
  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;
}

} // namespace naluUnit
}  // namespace sierra

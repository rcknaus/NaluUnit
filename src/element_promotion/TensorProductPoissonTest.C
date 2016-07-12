/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/TensorProductPoissonTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/MasterElement.h>
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/HigherOrderLaplacianQuad.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromotedElementIO.h>
#include <nalu_make_unique.h>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
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
//TensorProductPoissonTest - Use a four high-order elements to solve
// the "heat conduction MMS" to effectively floating point precision
//==========================================================================

#define POLY_ORDER 10
TensorProductPoissonTest::TensorProductPoissonTest(std::string meshName, bool printTiming)
  : meshName_(std::move(meshName)),
    order_(10),
    outputTiming_(true),
    timeMainLoop_(0.0),
    timeMetric_(0.0),
    timeLHS_(0.0),
    timeResidual_(0.0),
    timeGather_(0.0),
    timeVolumeMetric_(0.0),
    timeVolumeSource_(0.0),
    countAssemblies_(0),
    testTolerance_(1.0e-8), // 1.0e-8 is conservative even for the randomly perturbed case
    randomlyPerturbCoordinates_(true)
{
  // Nothing
}
//--------------------------------------------------------------------------
TensorProductPoissonTest::~TensorProductPoissonTest() = default;
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::execute()
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
  assemble_poisson<10>();
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

    NaluEnv::self().naluOutputP0() << "    -- Avg. Total element assembly: "
        << timeMainLoop_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. Gather: "
        << timeGather_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. Metric computation: "
        << timeMetric_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. LHS computation: "
        << timeLHS_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. Metric volume computation: "
        << timeVolumeMetric_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. Volumetric source computation: "
        << timeVolumeSource_ << std::endl;

    NaluEnv::self().naluOutputP0() << "        -- Avg. Residual computation: "
        << timeResidual_ << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to solve global matrix equation: "
        << timeSolver << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to update solution: "
        << timeUpdate << std::endl;

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
template <unsigned pOrder> void
TensorProductPoissonTest::assemble_poisson()
{
  // Poisson equation assembly algorithm for quadrilateral elements
  constexpr unsigned dim = 2;

  // set-up
  constexpr int nodesPerElement = (pOrder+1)*(pOrder+1);
  constexpr int lhsSize = nodesPerElement * nodesPerElement;

  // Operations
  auto func = MMSFunction(dim);
  auto laplaceOps = HigherOrderLaplacianQuad<pOrder>(*elem_);
  auto meSCS = HigherOrderQuad2DSCS(*elem_);

  std::vector<double> lhs(lhsSize, 0.0);
  std::vector<double> rhs(nodesPerElement,0.0);

  // scratch arrays
  std::array<size_t, nodesPerElement> indices{};
  std::array<double, nodesPerElement> scalar_field_data{};
  std::array<double, nodesPerElement*dim> coordinates{};
  std::array<double, nodesPerElement> nodalSource{};
  std::array<std::array<double, pOrder*(pOrder+1)>,4> metric_laplace{{}};
  std::array<double, nodesPerElement> metric_vol{};

  const auto& nodeMap = elem_->nodeMap;

  ThrowRequireMsg(elem_->useReducedGeometricBasis, "Only subparametric mapping for now");
  const double* const geomR = &meSCS.geometricShapeDerivs_[0];
  const double* const geomS = &meSCS.geometricShapeDerivs_[dim * pOrder * (pOrder+1) * 4];

  const auto& selector = stk::mesh::selectUnion(superPartVector_);
  const auto& buckets = bulkData_->get_buckets(stk::topology::ELEMENT_RANK, selector);

  // main loop
  countAssemblies_ = 0;
  for (const auto* ib : buckets) {
    const auto& b = *ib;
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      double timeMain = -MPI_Wtime();

      // zero arrays
      for (int p = 0; p < lhsSize; ++p) { lhs[p] = 0.0; }
      for (int p = 0; p < nodesPerElement; ++p) { rhs[p] = 0.0; }

       // gather data
      double timeGather = -MPI_Wtime();
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int vector_index = 0;
      for (int i = 0; i < nodesPerElement; ++i) {
        const double * coords = stk::mesh::field_data(*coordinates_, node_rels[i]);
        scalar_field_data[elem_->tensor_index(i)] = *stk::mesh::field_data(*q_, node_rels[i]);
        for (unsigned j=0; j < dim; ++j) {
          coordinates[vector_index] = coords[j];
          ++vector_index;
        }
      }
      timeGather += MPI_Wtime();
      timeGather_ += timeGather;

      // compute the metric for this element
      double timeMetric = -MPI_Wtime();
      laplaceOps.diffusion_metric(geomR, geomS, coordinates.data(), metric_laplace);
      timeMetric += MPI_Wtime();
      timeMetric_ += timeMetric;

      // compute left-hand side
      double timeLHS = -MPI_Wtime();
      laplaceOps.elemental_laplacian(metric_laplace, lhs.data());
      timeLHS += MPI_Wtime();
      timeLHS_ += timeLHS;

      // compute source term metric (det J)
      double timeVolumeMetric = -MPI_Wtime();
      double scv_error = 0.0;
      meSCV_->determinant(1, coordinates.data(), metric_vol.data(), &scv_error);
      timeVolumeMetric += MPI_Wtime();
      timeVolumeMetric_ += timeVolumeMetric;

      // compute volumetric source
      double timeVolumeSource = -MPI_Wtime();
      for (int i = 0; i < nodesPerElement; ++i) {
        nodalSource[i] = -func.exact_laplacian(&coordinates[nodeMap[i] * dim]) * metric_vol[i];
      }
      laplaceOps.volumetric_source(nodalSource.data(), rhs.data());
      timeVolumeSource += MPI_Wtime();
      timeVolumeSource_ += timeVolumeSource;

      // compute residual
      double timeRHS = -MPI_Wtime();
      laplaceOps.elemental_residual(metric_laplace, scalar_field_data.data(), rhs.data());
      timeRHS += MPI_Wtime();
      timeResidual_ += timeRHS;

      // sum into the global matrix -- not timed since this is just to test correctness
      // and the actual "sumInto" will be very different
      for (int j = 0; j < nodesPerElement; ++j) {
        indices[j] = rowMap_.at(node_rels[j]);
      }
      sum_into_global(indices.data(), lhs.data(), rhs.data(), nodesPerElement);

      timeMain += MPI_Wtime();
      timeMainLoop_ += timeMain;
      ++countAssemblies_;
    }
  }
  timeMainLoop_ /= static_cast<double>(countAssemblies_);
  timeMetric_ /= static_cast<double>(countAssemblies_);
  timeLHS_ /= static_cast<double>(countAssemblies_);
  timeResidual_ /= static_cast<double>(countAssemblies_);
  timeGather_ /= static_cast<double>(countAssemblies_);
  timeVolumeMetric_ /= static_cast<double>(countAssemblies_);
  timeVolumeSource_ /= static_cast<double>(countAssemblies_);
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::update_field()
{
  // update element boundaries
  auto selector =  stk::mesh::selectUnion(superPartVector_);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, selector);

  for (const auto* ib : node_buckets) {
    const auto& b = *ib;
    double* q = stk::mesh::field_data(*q_, b);
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      q[k] += delta_(rowMap_.at(b[k]));
    }
  }
}
//--------------------------------------------------------------------------
bool
TensorProductPoissonTest::check_solution()
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
TensorProductPoissonTest::setup_mesh()
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

  ThrowRequireMsg(metaData_->spatial_dimension() == 2, "Only 2D for now");
  elem_ = ElementDescription::create(metaData_->spatial_dimension(), order_, "SGL", true);
  ThrowRequire(elem_.get() != nullptr);
  meSCV_ = make_master_volume_element(*elem_);
  meSCS_ = make_master_subcontrol_surface_element(*elem_);

  setup_super_parts();
  register_fields();

  // populate bulk data
  ioBroker_->populate_bulk_data();

  if (randomlyPerturbCoordinates_) {
    perturb_coordinates(0.25, 0.15);
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
void
TensorProductPoissonTest::initialize_matrix()
{
  // count interior nodes
  const auto& node_buckets =
      bulkData_->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superPartVector_));

  // set-up connectivity
  size_t nodeNumber = 0;
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length   = b.size();
    for ( size_t k = 0 ; k < length ; ++k ) {
      rowMap_.insert({b[k], nodeNumber});
      ++nodeNumber;
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
void
TensorProductPoissonTest::apply_dirichlet()
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
void
TensorProductPoissonTest::solve_matrix_equation()
{
  Teuchos::SerialDenseSolver<int,double> solver;
  solver.setMatrix(Teuchos::rcp(&lhs_,false));
  solver.setVectors(Teuchos::rcp(&delta_,false), Teuchos::rcp(&rhs_,false));
  solver.equilibrateMatrix(); solver.equilibrateRHS();
  solver.solve();
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::sum_into_global(
  const size_t* indices,
  double* lhs_local,
  double* rhs_local,
  int length)
{
  for (int j = 0; j < length; ++j) {
    rhs_(indices[j]) += rhs_local[j];
    for (int i = 0; i < length; ++i) {
      lhs_(indices[j], indices[i]) += lhs_local[i+j*length];
    }
  }
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::output_banner()
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
  fineOutputName_   = "test_output/tensor" + elemType + ".e";

  NaluEnv::self().naluOutputP0()
      << "Using '" << elemType
      << "' Elements with tensor-product assembly to solve a Poisson equation MMS"
      <<   std::endl;

  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
TensorProductPoissonTest::make_master_volume_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCV>(elem);
  }
  return make_unique<HigherOrderHexSCV>(elem);
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
TensorProductPoissonTest::make_master_subcontrol_surface_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCS>(elem);
  }
  return make_unique<HigherOrderHexSCS>(elem);
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::register_fields()
{
  for (auto* basePart : originalPartVector_) {
    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_, *basePart, metaData_->spatial_dimension());

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

    q_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *superPart);

    qExact_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "exact_scalar"));
    stk::mesh::put_field(*qExact_, *superPart);
  }
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::setup_super_parts()
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
TensorProductPoissonTest::set_output_fields()
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
TensorProductPoissonTest::perturb_coordinates(double elem_size, double fac)
{
  std::mt19937 rng;
  rng.seed(0);
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
        coords[k * dim + j] += coeff(rng);
      }
    }
  }
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::initialize_fields()
{
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1,1);

  int dim = metaData_->spatial_dimension();
  auto func = MMSFunction(dim);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superPartVector_));
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length  = b.size();
    double* q = stk::mesh::field_data(*q_, b);
    double* qExact = stk::mesh::field_data(*qExact_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      q[k] = coeff(rng);
      qExact[k] = func.value(&coords[k*dim]);
    }
  }
}
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::output_results()
{
  output_result("Poisson", check_solution());
  promoteIO_->write_database_data(0.0);
  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;
}

} // namespace naluUnit
}  // namespace sierra

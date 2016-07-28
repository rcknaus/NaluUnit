/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/new_assembly/TensorProductPoissonTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/MasterElement.h>
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/new_assembly/HighOrderLaplacianQuad.h>
#include <element_promotion/new_assembly/HighOrderGeometry.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromotedElementIO.h>
#include <nalu_make_unique.h>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <TestHelper.h>
#include <KokkosInterface.h>

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
#include <stk_util/environment/CPUTime.hpp>
#include <stk_util/environment/perf_util.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <limits>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
//TensorProductPoissonTest - Use a four high-order elements to solve
// the "heat conduction MMS" to effectively floating point precision
//==========================================================================
TensorProductPoissonTest::TensorProductPoissonTest(
  std::string meshName,
  int order,
  bool printTiming)
  : meshName_(std::move(meshName)),
    order_(order),
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

  // number of runs for averaging timing data
  double timeAssembly = -MPI_Wtime();
  unsigned numRuns = outputTiming_ ? 1000 : 1;
  for (unsigned j = 0; j < numRuns; ++j) {
    lhs_.putScalar(0.0);
    rhs_.putScalar(0.0);
    assemble_poisson(order_);
  }
  timeAssembly += MPI_Wtime();

  double timeSolveAndUpdate = -MPI_Wtime();
  apply_dirichlet();
  solve_matrix_equation();
  update_field();
  timeSolveAndUpdate += MPI_Wtime();

  totalTime += MPI_Wtime();

  if (outputTiming_) {
    // average time
    timeMainLoop_ /= countAssemblies_;
    timeMetric_ /= countAssemblies_;
    timeLHS_ /= countAssemblies_;
    timeResidual_ /= countAssemblies_;
    timeGather_ /= countAssemblies_;
    timeVolumeMetric_ /= countAssemblies_;
    timeVolumeSource_ /= countAssemblies_;

    static const int NUM_TIMERS = 11;
    const double timers[NUM_TIMERS] = {
        timeSetup, timeAssembly,
        timeMainLoop_, timeGather_,
        timeMetric_, timeLHS_,
        timeVolumeMetric_, timeVolumeSource_,
        timeResidual_, timeSolveAndUpdate,  totalTime
    };
    const char* timer_names[NUM_TIMERS] = {
        "setup", "matrix assembly (run 1000 times)",
        "avg. element assembly", "avg. gather",
        "avg. surface metric computation", "avg. lhs assembly",
        "avg. volume metric computation", "avg. volumetric source computation",
        "avg. residual evaluation", "solve and update",
        "Total"
    };

    stk::print_timers(&timers[0], &timer_names[0], NUM_TIMERS);
  }
  output_results();
}
//--------------------------------------------------------------------------
struct MMSFunction {
  MMSFunction() : k(1.0), pi(std::acos(-1.0)) {};

  double exact_solution(double x, double y) {
    return (0.25*(std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y)));
  }

  double exact_laplacian(double x, double y) const
  {
    return ( -(k*pi)*(k*pi) * (std::cos(2.0*k*pi*x) + std::cos(2.0*k*pi*y)) );
  };

  double k;
  double pi;
};
//--------------------------------------------------------------------------
void
TensorProductPoissonTest::assemble_poisson(unsigned pOrder)
{
  switch (pOrder)
  {
    case  1: assemble_poisson< 1>(); break;
    case  2: assemble_poisson< 2>(); break;
    case  3: assemble_poisson< 3>(); break;
    case  4: assemble_poisson< 4>(); break;
    case  5: assemble_poisson< 5>(); break;
    case  6: assemble_poisson< 6>(); break;
    case  7: assemble_poisson< 7>(); break;
    case  8: assemble_poisson< 8>(); break;
    case  9: assemble_poisson< 9>(); break;
    case 10: assemble_poisson<10>(); break;
    case 15: assemble_poisson<15>(); break;
    default: throw std::runtime_error("Sorry, order " + std::to_string(pOrder) + " is not supported");
  }
}
//--------------------------------------------------------------------------
template <unsigned poly_order> void
TensorProductPoissonTest::assemble_poisson()
{
  // Poisson equation assembly algorithm for quadrilateral elements

  // definitions
  constexpr unsigned dim = 2;
  constexpr unsigned nodes1D = poly_order+1;
  constexpr unsigned nodesPerElement = (poly_order+1)*(poly_order+1);

  // Operations for assembly
  auto laplaceOps = HighOrderLaplacianQuad<poly_order>();
  auto geomOps = HighOrderGeometryQuad<poly_order>();

  // scratch
  typename QuadViews<poly_order>::matrix_array       lhs("left-hand side");
  typename QuadViews<poly_order>::nodal_scalar_array rhs("right-hand side");
  typename QuadViews<poly_order>::scs_tensor_array   metric_laplace("grid metric for laplacian");
  typename QuadViews<poly_order>::nodal_vector_array coordinates("element nodal coordinates");
  typename QuadViews<poly_order>::nodal_scalar_array metric_vol("Det(J) evaluated at the nodes");
  typename QuadViews<poly_order>::nodal_scalar_array scalar("scalar field data");
  typename QuadViews<poly_order>::nodal_scalar_array nodalSource("nodal source field");

  // map from default node-ordering to tensor product node-ordering
  const std::vector<unsigned>& nodeMap = elem_->nodeMap;

  const auto& buckets =
      bulkData_->get_buckets(stk::topology::ELEMENT_RANK, stk::mesh::selectUnion(superPartVector_));

  for (const auto* ib : buckets) {
    const auto& b = *ib;
    const auto length = b.size();
    for (size_t k = 0; k < length; ++k) {
      double timeMainStart = MPI_Wtime();

      // zero arrays
      Kokkos::deep_copy(lhs, 0.0);
      Kokkos::deep_copy(rhs, 0.0);

       // gather data
      double timeGatherStart = MPI_Wtime();
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      for (unsigned j = 0; j < nodes1D; ++j) {
        for (unsigned i = 0; i < nodes1D; ++i) {
          stk::mesh::Entity node = node_rels[nodeMap[i + nodes1D * j]];
          scalar(j, i) = *stk::mesh::field_data(*q_, node);
          nodalSource(j, i) = *stk::mesh::field_data(*source_, node);
          const double * coords = stk::mesh::field_data(*coordinates_, node);
          for (unsigned k = 0; k < dim; ++k) {
            coordinates(k, j, i) = coords[k];
          }
        }
      }
      timeGather_ += MPI_Wtime() - timeGatherStart;

      // compute the metric for this element
      double timeMetricStart = MPI_Wtime();
      geomOps.diffusion_metric_linear(coordinates, metric_laplace);
      timeMetric_ += MPI_Wtime() - timeMetricStart;

      // compute source term metric (det J)
      double timeVolumeMetricStart = MPI_Wtime();
      geomOps.volume_metric_linear(coordinates, metric_vol);
      timeVolumeMetric_ += MPI_Wtime() - timeVolumeMetricStart;

      // compute left-hand side
      double timeLHSStart = MPI_Wtime();
      laplaceOps.elemental_laplacian_matrix(metric_laplace, lhs);
      timeLHS_ += MPI_Wtime() - timeLHSStart;

      // compute volumetric source and add to rhs
      double timeVolumeSourceStart = MPI_Wtime();
      laplaceOps.volumetric_source(metric_vol, nodalSource, rhs);
      timeVolumeSource_ += MPI_Wtime() - timeVolumeSourceStart;

      // compute action of left-hand side and subtract from rhs to form residual
      double timeRHSStart = MPI_Wtime();
      laplaceOps.elemental_laplacian_action(metric_laplace, scalar, rhs);
      timeResidual_ += MPI_Wtime() - timeRHSStart;

      // sum into the global matrix -- not timed since this is just to test correctness
      // and the actual "sumInto" will be very different
      size_t indices[nodesPerElement];
      for (int j = 0; j < nodesPerElement; ++j) {
        indices[j] = rowMap_.at(node_rels[nodeMap[j]]);
      }
      sum_into_global(indices, lhs.data(), rhs.data(), nodesPerElement);

      timeMainLoop_ += MPI_Wtime() - timeMainStart;
      ++countAssemblies_;
    }
  }
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
      if (!std::isfinite(q[k])) {
        NaluEnv::self().naluOutputP0()
            << "Poisson test experienced a non-finite number at GID, " << bulkData_->identifier(b[k]) << " ";
        return false;
      }
      maxError = std::max(maxError, std::abs(q[k] - qExact[k]));
    }
  }

  // error should be small for high-enough order
  if (maxError >= testTolerance_) {
    NaluEnv::self().naluOutputP0()
        << "Poisson test failed with a maximum error of "
        << maxError << " vs a tolerance of "
        << testTolerance_ << ", ";
  }

  // test failed to iterate over any nodes at all ...
  // something went wrong with mesh creation
  if (maxError < 0 ) {
    NaluEnv::self().naluOutputP0()
        << "Poisson test failed to iterate over any nodes ";
  }
  return (maxError < testTolerance_ && maxError > 0);
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

  setup_super_parts();
  register_fields();

  // populate bulk data
  ioBroker_->populate_bulk_data();

  // perturb coordinates before promotion, so promoted mesh is not-curved
  if (randomlyPerturbCoordinates_) {
    perturb_coordinates(0.125, 0.1);
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
      bulkData_->get_buckets(
        stk::topology::NODE_RANK,
        stk::mesh::selectUnion(superPartVector_)
  );

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
  auto func = MMSFunction();

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
      rhs_(index) = func.exact_solution(coords[k*dim + 0 ], coords[k*dim+1]) - q[k];
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
      lhs_(indices[j], indices[i]) += lhs_local[i + length * j];
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
void
TensorProductPoissonTest::register_fields()
{
  coordinates_ =  &(metaData_-> declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates"));
  q_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
  source_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "source"));
  qExact_ = &(metaData_-> declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "exact_scalar"));

  const auto& allSuperParts = stk::mesh::selectUnion(superPartVector_);
  stk::mesh::put_field(*coordinates_, allSuperParts, metaData_->spatial_dimension());
  stk::mesh::put_field(*q_, allSuperParts);
  stk::mesh::put_field(*source_, allSuperParts);
  stk::mesh::put_field(*qExact_, allSuperParts);
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
          auto topo = metaData_->spatial_dimension() == 2 ?
              stk::create_superedge_topology(static_cast<unsigned>(elem_->nodesPerFace))
            : stk::create_superface_topology(static_cast<unsigned>(elem_->nodesPerFace));

          stk::mesh::Part* superFacePart = &metaData_->declare_part_with_topology(
            super_subset_part_name(subset->name(), elem_->nodesPerElement, elem_->nodesPerFace),
            topo
          );
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
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-fac*elem_size, fac*elem_size);

  auto selector = stk::mesh::selectUnion(originalPartVector_);
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, selector);
  int dim = metaData_->spatial_dimension();

  for (const auto ib : node_buckets ) {
    double* coords = stk::mesh::field_data(*coordinates_, *ib);
    for ( size_t k = 0 ; k < ib->size() ; ++k ) {
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
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-1,1);

  int dim = metaData_->spatial_dimension();
  auto func = MMSFunction();
  const auto& node_buckets =
      bulkData_->get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superPartVector_));
  for (const auto ib : node_buckets ) {
    const auto& b = *ib ;
    const auto length  = b.size();
    double* q = stk::mesh::field_data(*q_, b);
    double* qExact = stk::mesh::field_data(*qExact_, b);
    double* source = stk::mesh::field_data(*source_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( size_t k = 0 ; k < length ; ++k ) {
      q[k] = coeff(rng);
      qExact[k] = func.exact_solution(coords[k*dim+0], coords[k*dim+1]);
      source[k] = -func.exact_laplacian(coords[k*dim+0], coords[k*dim+1]);
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

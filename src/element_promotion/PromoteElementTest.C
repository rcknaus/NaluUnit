/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromoteElementTest.h>

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
#include <nalu_make_unique.h>
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

#include <Teuchos_BLAS.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// PromoteElementTests - a set of tests for element promotion
//==========================================================================
PromoteElementTest::PromoteElementTest(
  int dimension,
  int order,
  std::string meshName,
  std::string quadType)
  : activateAura_(false),
    currentTime_(0.0),
    resultsFileIndex_(1),
    meshName_(std::move(meshName)),
    defaultFloatingPointTolerance_(1.0e-12),
    constScalarField_(true),
    nDim_(dimension),
    order_(order),
    outputTiming_(false),
    quadType_(quadType)
{
}
//--------------------------------------------------------------------------
PromoteElementTest::~PromoteElementTest() = default;
//--------------------------------------------------------------------------
void
PromoteElementTest::execute()
{

  if(nDim_ == 2) {
    unsigned nodes = (order_+1)*(order_+1);
    elemType_ = "Quad" + std::to_string(nodes);
  }
  else {
    unsigned nodes = (order_+1)*(order_+1)*(order_+1);
    elemType_ = "Hex" + std::to_string(nodes);
  }
  coarseOutputName_ = "test_output/coarse_output/coarse_" + elemType_ + ".e";
  restartName_ = "test_output/" + elemType_ + ".rs";
  fineOutputName_   = "test_output/fine_output/fine_" + elemType_ + ".e";

  NaluEnv::self().naluOutputP0() << "Promoting to a '" << elemType_
                                 << "' Element with quadrature type '" << quadType_ << "' ..."
                                 <<   std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;

  elem_ = ElementDescription::create(nDim_, order_, quadType_);
  ThrowRequire(elem_ != nullptr);

  promoteElement_ = make_unique<PromoteElement>(*elem_);
  meSCV_ = create_master_volume_element(*elem_);
  meSCS_ = create_master_subcontrol_surface_element(*elem_);
  meBC_  = create_master_boundary_element(*elem_);

  auto timeA = MPI_Wtime();
  setup_mesh();
  auto timeB = MPI_Wtime();

  // save the original node count to check the assumption that no new nodes
  // are added to the original part vector
  unsigned originalNodes = count_nodes(stk::mesh::selectUnion(originalPartVector_));

  auto timeC = MPI_Wtime();
  bulkData_->modification_begin();
  promoteElement_->promote_elements(
    originalPartVector_,
    *coordinates_,
    *bulkData_
  );
  bulkData_->modification_end();
  auto timeD = MPI_Wtime();

  initialize_fields();

  auto timeE = MPI_Wtime();
  compute_dual_nodal_volume();
  auto timeF = MPI_Wtime();
  compute_projected_nodal_gradient();
  auto timeG = MPI_Wtime();

  output_result("DNV       ", check_dual_nodal_volume());
  output_result("PNG       ", check_projected_nodal_gradient());
  output_result("Node count", check_node_count(elem_->polyOrder, originalNodes));
  set_output_fields();
  output_results();
  auto timeH = MPI_Wtime();

  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Time to setup-mesh: "
        << timing_wall(timeA, timeB) << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to promote elements: "
        << timing_wall(timeC, timeD) << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to compute dual nodal volume: "
        << timing_wall(timeE, timeF) << std::endl;

    NaluEnv::self().naluOutputP0() << "Time to compute projected nodal gradient: "
        << timing_wall(timeF, timeG) << std::endl;

    NaluEnv::self().naluOutputP0() << "Total time for test: "
        << timing_wall(timeA, timeH) << std::endl;
  }

  NaluEnv::self().naluOutputP0() << "-------------------------"
      << std::endl;
}
//--------------------------------------------------------------------------
void
PromoteElementTest::setup_mesh()
{
  stk::ParallelMachine pm = NaluEnv::self().parallel_comm();

  //mesh setup
  metaData_ = make_unique<stk::mesh::MetaData>();
  auto aura = activateAura_ ?
      stk::mesh::BulkData::AUTO_AURA : stk::mesh::BulkData::NO_AUTO_AURA;
  bulkData_ = make_unique<stk::mesh::BulkData>(*metaData_, pm, aura);
  ioBroker_ = make_unique<stk::io::StkMeshIoBroker>(pm);
  ioBroker_->set_bulk_data(*bulkData_);

  // deal with input mesh
  ioBroker_->add_mesh_database(meshName_, stk::io::READ_MESH);
  ioBroker_->create_input_mesh();

  nDim_ = metaData_->spatial_dimension();

  register_fields();

  // populate bulk data
  ioBroker_->populate_bulk_data();
}
//--------------------------------------------------------------------------
double
PromoteElementTest::timing_wall(double timeA, double timeB)
{
  double maxTiming = 0;
  if (bulkData_->parallel_size() > 1) {
    double localTiming = timeB-timeA;
    stk::all_reduce_max(bulkData_->parallel(), &localTiming, &maxTiming, 1);
  }
  else {
    maxTiming = timeB-timeA;
  }
  return maxTiming;
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
PromoteElementTest::create_master_volume_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCV>(elem);
  }
  return make_unique<HigherOrderHexSCV>(elem);
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
PromoteElementTest::create_master_subcontrol_surface_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCS>(elem);
  }
  return make_unique<HigherOrderHexSCS>(elem);
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
PromoteElementTest::create_master_boundary_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderEdge2DSCS>(elem);
  }
  return make_unique<HigherOrderQuad3DSCS>(elem);
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_dual_nodal_volume()
{
  auto selector = stk::mesh::selectUnion(superElemPartVector_)
                & metaData_->locally_owned_part();

  if (quadType_ == "SGL") {
    compute_dual_nodal_volume_interior_SGL(selector);
  }
  else {
    compute_dual_nodal_volume_interior(selector);
  }

  if (bulkData_->parallel_size() > 1) {
    stk::mesh::parallel_sum(*bulkData_, {dualNodalVolume_});
    stk::mesh::parallel_sum(*bulkData_, {sharedElems_});
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient()
{
  auto interiorSelector = stk::mesh::selectUnion(superElemPartVector_)
                & metaData_->locally_owned_part();

  auto boundarySelector = stk::mesh::selectUnion(originalPartVector_)
                & metaData_->locally_owned_part();

  unsigned numRuns = 1;
  double totalTime = 0.0;
  timer_ = 0.0;
  for (unsigned runNumber = 0; runNumber < numRuns; ++runNumber) {
    //reset fields to the initial condition
    initialize_scalar();

    if (quadType_ == "SGL") {
      auto timeA = MPI_Wtime();
      compute_projected_nodal_gradient_interior_SGL(interiorSelector);
      compute_projected_nodal_gradient_boundary_SGL(boundarySelector);
      auto timeB = MPI_Wtime();

      totalTime += (timeB - timeA);
    }
    else {
      auto timeA = MPI_Wtime();
      compute_projected_nodal_gradient_interior(interiorSelector);
      compute_projected_nodal_gradient_boundary(boundarySelector);
      auto timeB = MPI_Wtime();

      totalTime += (timeB - timeA);
    }
  }


  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Average time for PNG calculation: "
        << totalTime / numRuns << std::endl;

    NaluEnv::self().naluOutputP0() << "Average time for core PNG calculation: "
        << timer_ / numRuns << std::endl;
  }

  if (bulkData_->parallel_size() > 1) {
    stk::mesh::parallel_sum(*bulkData_, {dqdx_});
  }
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_node_count(unsigned polyOrder, unsigned originalNodeCount)
{
  // check that the mesh is decomposed uniformly
  unsigned numProcs = bulkData_->parallel_size();
  unsigned b = std::pow(numProcs,1.0/nDim_);
  if (std::pow(static_cast<double>(b),nDim_)   != numProcs
   && std::pow(static_cast<double>(b+1),nDim_) != numProcs) {
    std::cout <<
        "Warning: Node count test assumes that the mesh is uniform and uniformly decomposed.  "
        "Test will definitely fail."
        << std::endl;
  }

  // test node count
  unsigned allNodes = count_nodes(metaData_->universal_part());
  unsigned totalNodes;
  if (nDim_ == 2) {
    totalNodes = std::pow(polyOrder*(static_cast<int>(std::sqrt(originalNodeCount+1))-1)+1,2);
  }
  else {
    totalNodes = std::pow(polyOrder*(static_cast<int>(std::cbrt(originalNodeCount+1))-1)+1,3);
  }

  if (allNodes != totalNodes) {
    NaluEnv::self().naluOutputP0() << "all nodes " << allNodes << " total Nodes " << totalNodes << std::endl;
    return false;
  }

  auto addedNodes = count_nodes(stk::mesh::selectUnion(promotedPartVector_));
  return addedNodes == totalNodes-originalNodeCount;
}
//--------------------------------------------------------------------------
size_t
PromoteElementTest::count_nodes(stk::mesh::Selector selector)
{
  size_t nodeCount = 0;
  const auto& node_buckets = bulkData_->get_buckets(stk::topology::NODE_RANK, selector);
  for (const auto* ib : node_buckets ) {
    const stk::mesh::Bucket& b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    nodeCount += length;
  }
  return nodeCount;
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_dual_nodal_volume_interior(
  stk::mesh::Selector& selector)
{
  MasterElement& masterElement = *meSCV_;

  // extract master element specifics
  const int nodesPerElement = masterElement.nodesPerElement_;
  const int numScvIp = masterElement.numIntPoints_;
  const int* ipNodeMap = masterElement.ipNodeMap();

  // define scratch field
  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK,
    selector);
  std::vector<double> ws_coordinates(nodesPerElement * nDim_);
  std::vector<double> ws_scv_volume(numScvIp);

  unsigned numRuns = 1;
  double totalTime = 0.0;

  for (unsigned runNumber = 0; runNumber < numRuns; ++runNumber) {
    //reset the state
    initialize_fields();

    auto timeA = MPI_Wtime();
    for (const auto* ib : elem_buckets) {
      const stk::mesh::Bucket & b = *ib;
      const stk::mesh::Bucket::size_type length = b.size();
      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        stk::mesh::Entity const* node_rels = b.begin_nodes(k);
        for (int ni = 0; ni < nodesPerElement; ++ni) {
          const stk::mesh::Entity node = node_rels[ni];
          const double* const coords = static_cast<double*>(stk::mesh::field_data(
            *coordinates_, node_rels[ni]));
          const int offSet = ni * nDim_;
          for (unsigned j = 0; j < nDim_; ++j) {
            ws_coordinates[offSet + j] = coords[j];
          }
          *stk::mesh::field_data(*sharedElems_, node) =
              promoteElement_->num_elems(node);
        }

        // compute integration point volume
        double scv_error = -1.0;
        masterElement.determinant(1, &ws_coordinates[0], &ws_scv_volume[0],
          &scv_error);
        ThrowRequireMsg(scv_error < 0.5, "Problem with determinant.");

        // assemble dual volume while scattering ip volume
        for (int ip = 0; ip < numScvIp; ++ip) {
          *stk::mesh::field_data(*dualNodalVolume_, node_rels[ipNodeMap[ip]]) +=
              ws_scv_volume[ip];
        }
      }
    }
    auto timeB = MPI_Wtime();
    totalTime += (timeB-timeA);
  }
  if (outputTiming_) {
    NaluEnv::self().naluOutputP0() << "Average time for DNV calculation: "
        << totalTime / numRuns << std::endl;
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_dual_nodal_volume_interior_SGL(
  stk::mesh::Selector& selector)
{

  MasterElement& masterElement = *meSCV_;

  // extract master element specifics
  const int nodesPerElement = masterElement.nodesPerElement_;
  const int numScvIp = masterElement.numIntPoints_;
  ThrowRequire(numScvIp == nodesPerElement);

  const int* ipNodeMap = masterElement.ipNodeMap();

  // define scratch field
  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  std::vector<double> ws_coordinates(nodesPerElement*nDim_);
  std::vector<double> ws_scv_volume(numScvIp);

  auto blas = Teuchos::BLAS<int, double>();
  std::vector<double> dnvTensor(nodesPerElement,0.0);
  std::vector<double> temp(nodesPerElement,0.0);
  auto quadOp = SGLQuadratureOps(*elem_);

  std::vector<double> dnvVector(nodesPerElement,0.0); // forward mapped dnv
  std::vector<double> dnvResult(nodesPerElement,0.0); // result

  unsigned numRuns = 1;
  double totalTime = 0.0;

  for (unsigned runNumber  = 0; runNumber < numRuns; ++runNumber) {
    //reset the state
    initialize_fields();

    auto timeA = MPI_Wtime();
    for (const auto* ib : elem_buckets ) {
      const stk::mesh::Bucket & b = *ib ;
      const stk::mesh::Bucket::size_type length = b.size();
      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        stk::mesh::Entity const* node_rels = b.begin_nodes(k);

        for (int ni = 0; ni < nodesPerElement; ++ni) {
          const stk::mesh::Entity node = node_rels[ni];
          const double* const coords = static_cast<double*>(stk::mesh::field_data(*coordinates_, node));
          const int offSet = ni*nDim_;
          for ( unsigned j = 0; j < nDim_; ++j ) {
            ws_coordinates[offSet+j] = coords[j];
          }
          *stk::mesh::field_data(*sharedElems_, node) = promoteElement_->num_elems(node);
        }

        // compute integration point volume
        double scv_error = -1.0;
        masterElement.determinant(1, &ws_coordinates[0], &ws_scv_volume[0], &scv_error);
        ThrowRequireMsg(scv_error < 0.5, "Problem with determinant.");

        if (nDim_ == 2) {
          /*
           * Currently, the 2D algorithm doesn't require a map stage in the gather
           * because the underlying ips are in the correct assumed order.  The usual mapping
           * is required for the scatter. The 3D algorithm has the reverse behavior: a mapping
           * is needed for the gather but not for the scatter
           */

          quadOp.volume_2D(ws_scv_volume.data(), dnvTensor.data());

          for (int ni = 0; ni < nodesPerElement; ++ni) {
            *stk::mesh::field_data(*dualNodalVolume_, node_rels[ipNodeMap[ni]]) += dnvTensor[ni];
          }
        }
        else {
          for (int ni = 0; ni < nodesPerElement; ++ni) {
            dnvVector[ipNodeMap[ni]] = ws_scv_volume[ni];
          }

          quadOp.volume_3D(dnvVector.data(), dnvResult.data());

          for (int ni = 0; ni < nodesPerElement; ++ni) {
            *stk::mesh::field_data(*dualNodalVolume_, node_rels[ni]) += dnvResult[ni];
          }
        }
      }
    }
    auto timeB = MPI_Wtime();
    totalTime += (timeB-timeA);
  }

  if (outputTiming_) {
        NaluEnv::self().naluOutputP0() << "Average time for DNV calculation: "
            << totalTime / numRuns << std::endl;
   }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_interior(
  stk::mesh::Selector& selector)
{
  auto timeA = MPI_Wtime();
  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    int dimension = meSCS_->nDim_;
    auto numScsIp = meSCS_->numIntPoints_;
    auto nodesPerElement = meSCS_->nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerElement);
    std::vector<double> ws_dualVolume(nodesPerElement);
    std::vector<double> ws_coords(nDim_*nodesPerElement);
    std::vector<double> ws_areav(nDim_*numScsIp);
    std::vector<double> ws_dqdx(nDim_*numScsIp);
    const auto* lrscv = meSCS_->adjacentNodes();
    std::vector<double> ws_shape_function(nodesPerElement*numScsIp);
    meSCS_->shape_fcn(ws_shape_function.data());

    for (size_t k = 0; k < length; ++k) {

      const auto* node_rels = b.begin_nodes(k);

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        stk::mesh::Entity node = node_rels[ni];

        const double * coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        ws_scalar[ni]     = *stk::mesh::field_data(*q_, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume_, node);

        // gather vectors
        const int offSet = ni*dimension;
        for ( int j=0; j < dimension; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meSCS_->determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        double* gradQL = stk::mesh::field_data(*dqdx_, node_rels[il]);
        double* gradQR = stk::mesh::field_data(*dqdx_, node_rels[ir]);

        double qIp = 0.0;
        const int offSet = ip*nodesPerElement;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
        }

        double inv_volL = 1.0/ws_dualVolume[il];
        double inv_volR = 1.0/ws_dualVolume[ir];

        for ( int j = 0; j < dimension; ++j ) {
          double fac = qIp*ws_areav[ip*dimension+j];
          gradQL[j] += fac*inv_volL;
          gradQR[j] -= fac*inv_volR;
        }
      }

    }
  }
  auto timeB = MPI_Wtime();
  timer_ += (timeB-timeA);
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_boundary(
  stk::mesh::Selector& selector)
{
  const auto& side_buckets =
      bulkData_->get_buckets(metaData_->side_rank(), selector);

  int dimension = meBC_->nDim_;
  for (const auto* ib : side_buckets ) {
    const stk::mesh::Bucket& b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    auto numScsIp = meBC_->numIntPoints_;
    auto nodesPerFace = meBC_->nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerFace);
    std::vector<double> ws_dualVolume(nodesPerFace);
    std::vector<double> ws_coords(nDim_*nodesPerFace);
    std::vector<double> ws_areav(nDim_*numScsIp);
    std::vector<double> ws_dqdx(nDim_*numScsIp);
    const auto* ipNodeMap = meBC_->ipNodeMap();

    std::vector<double> ws_shape_function(nodesPerFace*numScsIp);
    meBC_->shape_fcn(ws_shape_function.data());

    for (size_t k = 0; k < length; ++k) {
      //TODO(rcknaus): still need to route around stk for boundary contributions
      const auto* face_node_rels = promoteElement_->begin_side_nodes_all(b, k);

      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];

        const double * coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        ws_scalar[ni]     = *stk::mesh::field_data(*q_, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume_, node);

        // gather vectors
        const int offSet = ni*dimension;
        for ( int j=0; j < dimension; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
       }

      double scs_error = 0.0;
      meBC_->determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *gradQNN = stk::mesh::field_data(*dqdx_, nodeNN);
        double volNN = *stk::mesh::field_data(*dualNodalVolume_, nodeNN);

        // interpolate to scs point; operate on saved off ws_field
        double qIp = 0.0;
        const int offSet = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          qIp += ws_shape_function[offSet+ic]*ws_scalar[ic];
        }

        // nearest node volume
        double inv_volNN = 1.0/volNN;

        // assemble to nearest node
        for ( int j = 0; j < dimension; ++j ) {
          double fac = qIp*ws_areav[ip*dimension+j];
          gradQNN[j] += fac*inv_volNN;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_interior_SGL(stk::mesh::Selector& selector)
{
  unsigned numLines = nDim_*elem_->polyOrder;
  unsigned nodes1D = elem_->nodes1D;
  int ipsPerFace = (nDim_ == 2) ? nodes1D : nodes1D*nodes1D;

  int dimension = meSCS_->nDim_;
  auto quadOp = SGLQuadratureOps{*elem_};

  auto timeA = MPI_Wtime();
  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    auto numScsIp = meSCS_->numIntPoints_;
    auto nodesPerElement = meSCS_->nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerElement);
    std::vector<double> ws_dualVolume(nodesPerElement);
    std::vector<double> ws_coords(nDim_*nodesPerElement);
    std::vector<double> ws_areav(nDim_*numScsIp);
    std::vector<double> ws_dqdx(nDim_*numScsIp);
    const auto* lrscv = meSCS_->adjacentNodes();
    std::vector<double> ws_shape_function(nodesPerElement*numScsIp);
    meSCS_->shape_fcn(ws_shape_function.data());

    //in contrast to other routines, things are easiest if this
    // is ordered dimension-by-dimension, e.g. all x-components first
    std::vector<double> integrand(numScsIp*nDim_);
    std::vector<double> integrated_result(numScsIp*nDim_);

    for (size_t k = 0; k < length; ++k) {
      const auto* node_rels = b.begin_nodes(k);

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        stk::mesh::Entity node = node_rels[ni];

        const double* coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        ws_scalar[ni]     = *stk::mesh::field_data(*q_, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume_, node);

        // gather vectors
        const int offSet = ni*dimension;
        for ( int j=0; j < dimension; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
      }

      double scs_error = 0.0;
      meSCS_->determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      int ipNodeOffset = 0;
      for (int ip = 0; ip < numScsIp; ++ip) {
        double qIp = 0.0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          qIp += ws_shape_function[ipNodeOffset]*ws_scalar[ic];
          ++ipNodeOffset;
        }

        int ip_offset = ip*dimension;
        for ( int j = 0; j < dimension; ++j ) {
          integrand[ip+j*numScsIp] = qIp*ws_areav[ip_offset+j];
        }
      }

      /*
       * Intercept before scatter, then apply special weighting procedure
       * grab all ips associated with a line
       * this relies on IPs being numbered line-by-line in the
       * Master Element routine
       */
      int line_offset = 0;
      for (unsigned j = 0; j < nDim_; ++j) {
        for (auto lineNumber = 0u; lineNumber < numLines; ++lineNumber) {

          if (nDim_ == 2) {
            quadOp.surface_2D(integrand.data(), integrated_result.data(), line_offset);
          }
          else {
            quadOp.surface_3D(integrand.data(), integrated_result.data(), line_offset);
          }
          line_offset += ipsPerFace;
        }
      }

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        double* gradQL = stk::mesh::field_data(*dqdx_, node_rels[il]);
        double* gradQR = stk::mesh::field_data(*dqdx_, node_rels[ir]);

        double inv_volL = 1.0/ws_dualVolume[il];
        double inv_volR = 1.0/ws_dualVolume[ir];

        for ( int j = 0; j < dimension; ++j ) {
          double fac = integrated_result[ip+j*numScsIp];
          gradQL[j] += fac*inv_volL;
          gradQR[j] -= fac*inv_volR;
        }
      }
    }
  }
  auto timeB = MPI_Wtime();
  timer_ += (timeB-timeA);
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_boundary_SGL(
  stk::mesh::Selector& selector)
{
  const auto& side_buckets =
      bulkData_->get_buckets(metaData_->side_rank(), selector);

  int ipsPerFace = (nDim_ == 2) ? elem_->nodes1D : elem_->nodes1D*elem_->nodes1D;

  int dimension = meBC_->nDim_;
  for (const auto* ib : side_buckets ) {
    const stk::mesh::Bucket& b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    auto numScsIp = meBC_->numIntPoints_;
    auto nodesPerFace = meBC_->nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerFace);
    std::vector<double> ws_dualVolume(nodesPerFace);
    std::vector<double> ws_coords(nDim_*nodesPerFace);
    std::vector<double> ws_areav(nDim_*numScsIp);
    std::vector<double> ws_dqdx(nDim_*numScsIp);
    const auto* ipNodeMap = meBC_->ipNodeMap();

    std::vector<double> ws_shape_function(nodesPerFace*numScsIp);
    meBC_->shape_fcn(ws_shape_function.data());

    std::vector<double> integrand(numScsIp*nDim_);
    std::vector<double> integrated_result(numScsIp*nDim_);

    auto quadOp = SGLQuadratureOps{*elem_};
    auto blas = Teuchos::BLAS<int, double>();
    int nodesPerElement = meBC_->nodesPerElement_;

    for (size_t k = 0; k < length; ++k) {
      //TODO(rcknaus): still need to route around stk for boundary contributions
      const auto* face_node_rels = promoteElement_->begin_side_nodes_all(b, k);

      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];

        const double * coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        ws_scalar[ni]     = *stk::mesh::field_data(*q_, node);
        ws_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume_, node);

        // gather vectors
        const int offSet = ni*dimension;
        for ( int j=0; j < dimension; ++j ) {
          ws_coords[offSet+j] = coords[j];
        }
       }

      double scs_error = 0.0;
      meBC_->determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

      int ipNodeOffset = 0;
      for (int ip = 0; ip < numScsIp; ++ip) {
        double qIp = 0.0;
        for (int ic = 0; ic < nodesPerElement; ++ic) {
          qIp += ws_shape_function[ipNodeOffset]*ws_scalar[ic];
          ++ipNodeOffset;
        }

        int ip_offset = ip*dimension;
        for ( int j = 0; j < dimension; ++j ) {
          integrand[ip+j*numScsIp] = qIp*ws_areav[ip_offset+j];
        }
      }

      int line_offset = 0;
      if (nDim_ == 2) {
        for (int j = 0; j < 2; ++j) {
          quadOp.surface_2D(integrand.data(), integrated_result.data(), line_offset);
          line_offset += ipsPerFace;
        }
      }
      else {
        for (int j = 0; j < 3; ++j) {
          quadOp.surface_3D(integrand.data(), integrated_result.data(), line_offset);
          line_offset += ipsPerFace;
        }
      }

      for (int ip = 0; ip < numScsIp; ++ip) {
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *gradQNN = stk::mesh::field_data(*dqdx_, nodeNN);
        double volNN = *stk::mesh::field_data(*dualNodalVolume_, nodeNN);

        // nearest node volume
        double inv_volNN = 1.0/volNN;

        // assemble to nearest node
        for ( int j = 0; j < dimension; ++j ) {
          gradQNN[j] += integrated_result[ip+j*numScsIp]*inv_volNN;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::dump_coords()
{
  stk::mesh::BucketVector const& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superElemPartVector_));
  for (const auto ib : elem_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const auto elem = b[k];
      stk::mesh::Entity const* node_rels = b.begin_nodes(k);

      std::cout << "Coords for elem " << bulkData_->identifier(elem) << ": ";
      for (size_t node = 0; node < elem_->nodes1D; ++node) {
        double* coords = stk::mesh::field_data(*coordinates_, node_rels[node]);
        if (nDim_ == 2 ) {

          std::cout << "("<< coords[0] << "," << coords[1] << ")" << " ";
        }
        else {
          std::cout << "(" << coords[0] << ","
                           << coords[1] << ","
                           << coords[2]  <<  ")" << " ";
        }
      }
      std::cout << std::endl;
    }
  }
}
//--------------------------------------------------------------------------
double
PromoteElementTest::determine_mesh_spacing()
{
  double meshSpacing = std::numeric_limits<double>::lowest();
  bool isUniform = true;
  auto selector = stk::mesh::selectUnion(originalPartVector_);
  stk::mesh::BucketVector const& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto ib : elem_buckets) {
    stk::mesh::Bucket& b = *ib;
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      stk::mesh::Entity const* node_rels = b.begin_nodes(k); // base nodes

      for (unsigned ni = 0; ni < 3; ++ni) {
        double* coords = stk::mesh::field_data(*coordinates_, node_rels[ni]);
        double* adjCoords = stk::mesh::field_data(*coordinates_, node_rels[ni + 1]);

        double dist = 0.0;
        for (unsigned j = 0; j < nDim_; ++j) {
          double d = coords[j] - adjCoords[j];
          dist += d * d;
        }
        dist = std::sqrt(dist);

        if (meshSpacing < 0.0) {
          meshSpacing = dist;
        }
        else if (!is_near(meshSpacing,dist, defaultFloatingPointTolerance_)) {
          isUniform = false;
        }

        if (nDim_ == 3) {
          // 4th node is diagonal from the 3rd node
          for (unsigned ni = 4; ni < 7; ++ni) {
            double* coords =
                stk::mesh::field_data(*coordinates_, node_rels[ni]);
            double* adjCoords =
                stk::mesh::field_data(*coordinates_, node_rels[ni + 1]);

            double dist = 0.0;
            for (unsigned j = 0; j < nDim_; ++j) {
              double d = coords[j] - adjCoords[j];
              dist += d * d;
            }

            dist = std::sqrt(dist);
            if (meshSpacing < 0.0) {
              meshSpacing = dist;
            }
            else if (meshSpacing != dist) {
              isUniform = false;
            }
          }
        }
      }
    }
  }
  if (!isUniform) {
    NaluEnv::self().naluOutputP0() <<
        "Warning: DNV and Node counts tests assume that the mesh is uniform.  Tests will definitely fail."
             << std::endl;
  }
  return meshSpacing;
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_projected_nodal_gradient()
{
  // test assumes that the scalar q is constant
  // will fail otherwise

  if (!constScalarField_) {
    NaluEnv::self().naluOutputP0()
        << "Warning: PNG test assumes a constant field.  Test will definitely fail."
        << std::endl;
  }
  bool testPassed = false;

  double tol = 1.0e-8; //P=5 is off on this test by 6.07e-10

  std::vector<double> zeroVec(nDim_,0.0);
  std::vector<double> ws_dqdx(nDim_);

  stk::mesh::Selector s_all_entities = metaData_->universal_part(); // all nodes
  stk::mesh::BucketVector const& node_buckets =
      bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
  for (const auto ib : node_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double* dqdx = stk::mesh::field_data(*dqdx_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      for (unsigned j = 0; j < nDim_; ++j) {
        ws_dqdx[j] = dqdx[k*nDim_+j];
      }

      if (is_near(ws_dqdx, zeroVec, tol)) {
        testPassed = true;
      }
      else {
        return false;
      }
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_dual_nodal_volume()
{
  if (nDim_ == 2) {
    return check_dual_nodal_volume_quad();
  }
  return check_dual_nodal_volume_hex();
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_dual_nodal_volume_quad()
{
  double meshSpacing = determine_mesh_spacing();

  auto scsEndLoc = elem_->quadrature->scsEndLoc();
  std::vector<double> exactDualNodalVolume(elem_->nodesPerElement);
  for (unsigned i = 0; i < elem_->nodes1D; ++i) {
    for (unsigned j = 0; j < elem_->nodes1D; ++j) {
      exactDualNodalVolume[elem_->tensor_product_node_map(i,j)] =
          0.25*(scsEndLoc[i+1]-scsEndLoc[i]) * (scsEndLoc[j+1]-scsEndLoc[j])
          * meshSpacing * meshSpacing;
    }
  }

  bool testPassed = false;

  stk::mesh::BucketVector const& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superElemPartVector_));
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity const* node_rels = b.begin_nodes(k);
      for (unsigned j = 0; j < b.num_nodes(k); ++j) {
        const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node_rels[j]);
        const auto num_elems = *stk::mesh::field_data(*sharedElems_, node_rels[j]);
        double exact = num_elems*exactDualNodalVolume[j];
        if (!is_near(dualNodalVolume,exact, defaultFloatingPointTolerance_)) {
          return false;
        }
        testPassed = true;
      }
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_dual_nodal_volume_hex()
{
  double meshSpacing = determine_mesh_spacing();

    auto scsEndLoc = elem_->quadrature->scsEndLoc();
    std::vector<double> exactDualNodalVolume(elem_->nodesPerElement);
    for (unsigned i = 0; i < elem_->nodes1D; ++i) {
      for (unsigned j = 0; j < elem_->nodes1D; ++j) {
        for (unsigned k = 0; k < elem_->nodes1D; ++k) {
        exactDualNodalVolume[elem_->tensor_product_node_map(i,j,k)] =
            0.125
            * (scsEndLoc[i+1]-scsEndLoc[i])
            * (scsEndLoc[j+1]-scsEndLoc[j])
            * (scsEndLoc[k+1]-scsEndLoc[k])
            * meshSpacing * meshSpacing * meshSpacing;
        }
      }
    }

    bool testPassed = false;

    stk::mesh::BucketVector const& elem_buckets =
        bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superElemPartVector_));
    for (const auto* ib : elem_buckets ) {
      const stk::mesh::Bucket & b = *ib ;
      const stk::mesh::Bucket::size_type length = b.size();
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        stk::mesh::Entity const* node_rels = b.begin_nodes(k);
        for (unsigned j = 0; j < b.num_nodes(k); ++j) {
          const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node_rels[j]);
          const auto num_elems = *stk::mesh::field_data(*sharedElems_, node_rels[j]);
          double exact = num_elems*exactDualNodalVolume[j];
          if (!is_near(dualNodalVolume,exact, defaultFloatingPointTolerance_)) {
            return false;
          }
          testPassed = true;
        }
      }
    }
    return testPassed;
}
//--------------------------------------------------------------------------
void
PromoteElementTest::register_fields()
{
  // extract blocks in the mesh with target names that are specified inline
  std::vector<std::string> targetNames;
  auto& meshParts = metaData_->get_mesh_parts();
  for (const auto* part : meshParts) {
    targetNames.push_back(part->name());
  }

  std::vector<std::string> promotedNames;

  // save space for parts of the input mesh
  for (auto& baseName : targetNames) {
    std::string promotedName = promote_part_name(baseName);

    stk::mesh::Part* targetPart = metaData_->get_part(baseName);

    stk::mesh::Part* superElemPart = nullptr;
    if (targetPart->topology().rank() == stk::topology::ELEM_RANK) {
      /* TODO(rcknaus): use this style after stk bug fix is pushed to Trilinos
       *
       *  superElemPart = &metaData_->declare_part_with_topology(
       *       super_element_part_name(targetName),
       *       stk::create_superelement_topology(static_cast<unsigned>(elem_->nodesPerElement))
       *  );
       */


      superElemPart = &metaData_->declare_part(
        super_element_part_name(baseName),
        stk::topology::ELEMENT_RANK
      );

      stk::mesh::set_topology(
        *superElemPart,
        stk::create_superelement_topology(static_cast<unsigned>(elem_->nodesPerElement))
      );


      stk::io::put_io_part_attribute(*superElemPart);

      superElemPartVector_.push_back(superElemPart);
    }

    stk::mesh::Part* promotedPart = &metaData_->declare_part(promotedName, stk::topology::NODE_RANK);


    // extract the parts
    originalPartVector_.push_back(targetPart);
    promotedPartVector_.push_back(promotedPart);

    // register nodal fields
    dualNodalVolume_ = &(metaData_->declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"));
    stk::mesh::put_field(*dualNodalVolume_, *targetPart);
    stk::mesh::put_field(*dualNodalVolume_, *promotedPart);
    if (superElemPart != nullptr) {
      stk::mesh::put_field(*dualNodalVolume_, *superElemPart);
    }

    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_,*targetPart, nDim_);
    stk::mesh::put_field(*coordinates_,*promotedPart, nDim_);
    if (superElemPart != nullptr) {
      stk::mesh::put_field(*coordinates_,*superElemPart, nDim_);
    }

    dqdx_ = &(metaData_->declare_field<VectorFieldType>(
       stk::topology::NODE_RANK, "dqdx"));
     stk::mesh::put_field(*dqdx_,*targetPart, nDim_);
     stk::mesh::put_field(*dqdx_,*promotedPart, nDim_);
     if (superElemPart != nullptr) {
       stk::mesh::put_field(*dqdx_, *superElemPart, nDim_);
     }

    // slightly helpful for checking dual nodal volume
    sharedElems_ = &(metaData_->declare_field<ScalarIntFieldType>(
      stk::topology::NODE_RANK, "elements_shared"));
    stk::mesh::put_field(*sharedElems_, *targetPart);
    stk::mesh::put_field(*sharedElems_, *promotedPart);
    if (superElemPart != nullptr) {
      stk::mesh::put_field(*sharedElems_, *superElemPart);
    }

    q_ = &(metaData_->
        declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *targetPart);
    stk::mesh::put_field(*q_, *promotedPart);
    if (superElemPart != nullptr) {
      stk::mesh::put_field(*q_, *superElemPart);
    }
   }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::set_output_fields()
{
  resultsFileIndex_ =
      ioBroker_->create_output_mesh(coarseOutputName_, stk::io::WRITE_RESULTS);

  restartFileIndex_ =
      ioBroker_->create_output_mesh(restartName_, stk::io::WRITE_RESTART);

  ioBroker_->add_field(resultsFileIndex_, *dualNodalVolume_, dualNodalVolume_->name());
  ioBroker_->add_field(restartFileIndex_, *dualNodalVolume_, dualNodalVolume_->name());

  promoteIO_ = make_unique<PromotedElementIO>(
    *elem_,
    *metaData_,
    *bulkData_,
    originalPartVector_,
    fineOutputName_
  );

  promoteIO_->add_fields({dualNodalVolume_, sharedElems_,q_,dqdx_});
}
//--------------------------------------------------------------------------
void
PromoteElementTest::initialize_fields()
{
  const double pi = std::acos(-1.0);

  stk::mesh::Selector s_all_entities = metaData_->universal_part();
  stk::mesh::BucketVector const& node_buckets =
      bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
  for (const auto ib : node_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double* dualNodalVolume = stk::mesh::field_data(*dualNodalVolume_, b);
    double* q = stk::mesh::field_data(*q_, b);
    double* dqdx = stk::mesh::field_data(*dqdx_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      dualNodalVolume[k] = 0.0;
      if (constScalarField_) {
        q[k] = 1.0;
      }
      else {
        if (nDim_ == 2) {
          q[k] = ( std::cos(2.0*pi*coords[0+k*nDim_])
                 + std::cos(2.0*pi*coords[1+k*nDim_]) ) * 0.25;
        }
        else {
          q[k] = ( std::cos(2.0*pi*coords[0+k*nDim_])
                 + std::cos(2.0*pi*coords[1+k*nDim_])
                 + std::cos(2.0*pi*coords[2+k*nDim_]) ) * 0.25;
        }
      }
      for (unsigned j =0; j < nDim_; ++j) {
        dqdx[j+k*nDim_] = 0.0;
      }
    }
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::initialize_scalar()
{
  const double pi = std::acos(-1.0);

  stk::mesh::Selector s_all_entities = metaData_->universal_part();
  stk::mesh::BucketVector const& node_buckets =
      bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
  for (const auto ib : node_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double* q = stk::mesh::field_data(*q_, b);
    double* dqdx = stk::mesh::field_data(*dqdx_, b);
    double* coords = stk::mesh::field_data(*coordinates_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      if (constScalarField_) {
        q[k] = 1.0;
      }
      else {
        if (nDim_ == 2) {
          q[k] = ( std::cos(2.0*pi*coords[0+k*nDim_])
                 + std::cos(2.0*pi*coords[1+k*nDim_]) ) * 0.25;
        }
        else {
          q[k] = ( std::cos(2.0*pi*coords[0+k*nDim_])
                 + std::cos(2.0*pi*coords[1+k*nDim_])
                 + std::cos(2.0*pi*coords[2+k*nDim_]) ) * 0.25;
        }
      }
      for (unsigned j =0; j < nDim_; ++j) {
        dqdx[j+k*nDim_] = 0.0;
      }
    }
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::output_results()
{
  ioBroker_->process_output_request(resultsFileIndex_, currentTime_);
  ioBroker_->process_output_request(restartFileIndex_, currentTime_);
  promoteIO_->write_database_data(currentTime_);
}
//--------------------------------------------------------------------------
std::string
PromoteElementTest::output_coords(stk::mesh::Entity node, unsigned dim)
{
  double* coords = stk::mesh::field_data(*coordinates_,node);

  std::string msg = "(" + std::to_string(coords[0]) + ", " + std::to_string(coords[1]) + ")";
  if (dim == 3) {
    msg = "(" + std::to_string(coords[0])
       + ", " + std::to_string(coords[1])
       + ", " + std::to_string(coords[2]) +")";
  }
  return msg;
}


} // namespace naluUnit
}  // namespace sierra

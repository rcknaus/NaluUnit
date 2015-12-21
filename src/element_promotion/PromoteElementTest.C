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
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <nalu_make_unique.h>

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

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// PromoteElement - unit test for PromoteElement
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PromoteElementTest::PromoteElementTest(std::string elemType, std::string meshName)
  : activateAura_(false),
    currentTime_(0.0),
    resultsFileIndex_(1),
    elemType_(elemType),
    meshName_(meshName),
    coarseOutputName_("test_output/coarse_output/coarse_" + elemType + ".e"),
    fineOutputName_("test_output/fine_output/fine_" + elemType + ".e"),
    floatingPointTolerance_(1.0e-11),
    constScalarField_(true)
{
}
//--------------------------------------------------------------------------
PromoteElementTest::~PromoteElementTest() {}
//--------------------------------------------------------------------------
void
PromoteElementTest::execute()
{
  NaluEnv::self().naluOutputP0() << "Welcome to the PromoteElement unit test" << std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;
  NaluEnv::self().naluOutputP0() << "Promoting to a " << elemType_ << " Element ..." << std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;

  std::map<bool, std::string> passFail = { {false,"FAIL"}, {true, "PASS"} };

  elem_ = ElementDescription::create(elemType_);
  ThrowRequire(elem_ != nullptr);

  if (elemType_ == "Quad9") {
    std::vector<unsigned> nodeMap =
    {
        0, 4, 1,
        7, 8, 5,
        3, 6, 2
    };
    ThrowRequire(elem_->nodeMap == nodeMap);
  }
  else if (elemType_ == "Quad16")
  {
    std::vector<unsigned> nodeMap =
      {
          0, 4, 5, 1,
          11, 12, 13, 6,
          10, 14, 15, 7,
          3, 9, 8, 2
      };
    ThrowRequire(elem_->nodeMap == nodeMap);
  }

  promoteElement_ = make_unique<PromoteElement>(*elem_);
  ThrowRequire(promoteElement_ != nullptr);

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
    *bulkData_,
    promotedPartVector_
  );
  bulkData_->modification_end();
  auto timeD = MPI_Wtime();

  bool interpCheck = false;
  bool derivCheck = false;
  bool quadCheck = false;
  bool pngCheck = false;
  if (nDim_ == 2) {
    interpCheck = check_interpolation();
    derivCheck = check_derivative();
    quadCheck = check_volume_quadrature();
    pngCheck = check_projected_nodal_gradient();
  }
  bool nodeCountCheck = check_node_count(elem_->polyOrder, originalNodes);

  initialize_fields();

  auto timeE = MPI_Wtime();
  compute_dual_nodal_volume();
  auto timeF = MPI_Wtime();
  compute_projected_nodal_gradient();
  auto timeG = MPI_Wtime();

  bool dnvCheck = check_dual_nodal_volume();
  set_output_fields();
  output_results();

  auto timeH = MPI_Wtime();

  NaluEnv::self().naluOutputP0() << "Time to setup-mesh: "
      << timing_wall(timeA,timeB) << std::endl;
  NaluEnv::self().naluOutputP0() << "Time to promote elements: "
      << timing_wall(timeC,timeD) << std::endl;
  NaluEnv::self().naluOutputP0() << "Time to compute dual nodal volume: "
      << timing_wall(timeE,timeF) << std::endl;
  if (nDim_ == 2) {
    NaluEnv::self().naluOutputP0() << "Time to compute projected nodal gradient: "
        << timing_wall(timeF,timeG) << std::endl;
  }
  NaluEnv::self().naluOutputP0() << "Total time for test: "
      << timing_wall(timeA,timeH) << std::endl;

  NaluEnv::self().naluOutputP0() << "-------------------------"
      << std::endl;

  if (nDim_ == 2) {
    NaluEnv::self().naluOutputP0() << "Interpolation test: "
        <<  passFail[interpCheck] << std::endl;
    NaluEnv::self().naluOutputP0() << "Derivative test: "
        <<  passFail[derivCheck] << std::endl;
    NaluEnv::self().naluOutputP0() << "Quadrature test: "
        <<  passFail[quadCheck] << std::endl;
    NaluEnv::self().naluOutputP0() << "Projected Nodal Gradient test: "
            <<  passFail[pngCheck] << std::endl;
  }

  NaluEnv::self().naluOutputP0() << "Node count test: "
      << passFail[nodeCountCheck] << std::endl;
  NaluEnv::self().naluOutputP0() << "Dual Nodal Volume test: "
       << passFail[dnvCheck] << std::endl;

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
void
PromoteElementTest::compute_dual_nodal_volume()
{
  auto selector = stk::mesh::selectUnion(originalPartVector_)
                & metaData_->locally_owned_part();
  if (nDim_ == 2) {
    compute_dual_nodal_volume_interior(
      HigherOrderQuad2DSCV{*elem_}, selector
    );
  }
  else {
    compute_dual_nodal_volume_interior(Hex27SCV{}, selector);
  }
}

//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient()
{
  auto selector = stk::mesh::selectUnion(originalPartVector_)
                & metaData_->locally_owned_part();
  if (nDim_ == 2) {
    compute_projected_nodal_gradient_interior(
      HigherOrderQuad2DSCS{*elem_}, selector
    );

    compute_projected_nodal_gradient_boundary(
      HigherOrderEdge2DSCS{*elem_}, selector
    );
  }
  else {
    //wip
  }

  if (bulkData_->parallel_size() > 1) {
    stk::mesh::parallel_sum(*bulkData_, {dqdx_});
  }
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_interpolation()
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and interpolate to a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;
  unsigned numIps = 1000;
  unsigned numTrials = 1000;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);
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

    if (is_near(approxInterp, exactInterp)) {
      testPassed = true;
    }
    else {
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_derivative()
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and compute derivatives at a vector of random points
  // (between -1.05 and 1.05 to check edges)
  bool testPassed = false;
  unsigned numIps = 1000;
  unsigned numTrials = 1000;


  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0,10.0);
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

    if (is_near(approxDeriv, exactDeriv)) {
      testPassed = true;
    }
    else {
      return false;
    }
  }
  return testPassed;
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_volume_quadrature()
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes
  bool testPassed = false;
  unsigned numTrials = 1000;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);
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
    for (int ip = 0; ip < masterElement.numIntPoints_; ++ip) {
      double interpValue = 0.0;
      for (unsigned nodeNumber = 0; nodeNumber < elem_->nodesPerElement; ++nodeNumber) {
        interpValue += interpWeights[ip*elem_->nodesPerElement+nodeNumber] * nodalValues[nodeNumber];
      }
      approxInt[ipNodeMap[ip]] +=  ipWeights[ip]*interpValue;
    }

    if (is_near(approxInt, exactInt)) {
      testPassed = true;
    }
    else {
      return false;
    }
  }

  return testPassed;
}
//--------------------------------------------------------------------------
double
PromoteElementTest::poly_val(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j);
  }
  return val;
}
//--------------------------------------------------------------------------
double
PromoteElementTest::poly_der(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j-1)*j;
  }
  return val;
}
//--------------------------------------------------------------------------
double
PromoteElementTest::poly_int(std::vector<double> coeffs,
  double xlower, double xupper)
{
  double upper = 0.0; double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j]*std::pow(xupper,j+1)/(j+1.0);
    lower += coeffs[j]*std::pow(xlower,j+1)/(j+1.0);
  }
  return (upper-lower);
}
//--------------------------------------------------------------------------
bool
PromoteElementTest::check_node_count(unsigned polyOrder, unsigned originalNodeCount)
{
  // check that the mesh is decomposed uniformly
  unsigned numProcs = bulkData_->parallel_size();
  unsigned b = std::pow(numProcs,1.0/nDim_);
  if (std::pow(static_cast<double>(b),nDim_) != numProcs
      && std::pow(static_cast<double>(b+1),nDim_) != numProcs) {
    std::cout <<
        "Warning: Test assumes a uniform mesh "
        "decomposition and will definitely fail the node count test otherwise"
        << std::endl;
  }

  // test node count
  unsigned allNodes = count_nodes(metaData_->universal_part());
  unsigned totalNodes;
  if (nDim_ == 2) {
    totalNodes = std::pow(polyOrder*(static_cast<int>(std::sqrt(originalNodeCount+1))-1)+1,2);
  }
  else {
    totalNodes = std::pow(2*std::cbrt(originalNodeCount)-1,3);
  }

  if (allNodes != totalNodes) {
    std::cout << "all nodes " << allNodes << " total Nodes " << totalNodes << std::endl;
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
  MasterElement&& masterElement,
  stk::mesh::Selector& selector)
{
  // extract master element specifics
  const int nodesPerElement = masterElement.nodesPerElement_;
  const int numScvIp = masterElement.numIntPoints_;
  const int* ipNodeMap = masterElement.ipNodeMap();

  // define scratch field
  std::vector<double> ws_coordinates(nodesPerElement*nDim_);
  std::vector<double> ws_scv_volume(numScvIp);

  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity const* node_rels = promoteElement_->begin_nodes_all(b,k);

      for (int ni = 0; ni < nodesPerElement; ++ni) {
        const stk::mesh::Entity node = node_rels[ni];
        const double* const coords = static_cast<double*>(stk::mesh::field_data(*coordinates_, node_rels[ni]));
        const int offSet = ni*nDim_;
        for ( unsigned j=0; j < nDim_; ++j ) {
          ws_coordinates[offSet+j] = coords[j];
        }
        *stk::mesh::field_data(*sharedElems_, node) = promoteElement_->num_elems(node);
      }

      // compute integration point volume
      double scv_error = -1.0;
      masterElement.determinant(1, &ws_coordinates[0], &ws_scv_volume[0], &scv_error);
      ThrowRequireMsg(scv_error < 0.5, "Problem with determinant.");

      // assemble dual volume while scattering ip volume
      for ( int ip = 0; ip < numScvIp; ++ip ) {
        *stk::mesh::field_data(*dualNodalVolume_, node_rels[ipNodeMap[ip]])
                 += ws_scv_volume[ip];
      }
    }
  }

  if (bulkData_->parallel_size() > 1) {
    stk::mesh::parallel_sum(*bulkData_, {dualNodalVolume_});
    stk::mesh::parallel_sum(*bulkData_, {sharedElems_});
  }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_interior(
  HigherOrderQuad2DSCS&& meSCS,
  stk::mesh::Selector& selector)
{
  const auto& elem_buckets = bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    int dimension = meSCS.nDim_;
    auto numScsIp = meSCS.numIntPoints_;
    auto nodesPerElement = meSCS.nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerElement);
    std::vector<double> ws_dualVolume(nodesPerElement);
    std::vector<double> ws_coords(2*nodesPerElement);
    std::vector<double> ws_areav(2*numScsIp);
    std::vector<double> ws_dqdx(2*numScsIp);
    const auto* lrscv = meSCS.adjacentNodes();

    for (size_t k = 0; k < length; ++k) {
      const auto& ws_shape_function = meSCS.shapeFunctions_;
      const auto* node_rels = promoteElement_->begin_nodes_all(b, k);

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
      meSCS.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

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
}
//--------------------------------------------------------------------------
void
PromoteElementTest::compute_projected_nodal_gradient_boundary(
  HigherOrderEdge2DSCS&& meSCS,
  stk::mesh::Selector& selector)
{
  const auto& side_buckets =
      bulkData_->get_buckets(metaData_->side_rank(), selector);

  int dimension = meSCS.nDim_;
  for (const auto* ib : side_buckets ) {
    const stk::mesh::Bucket& b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();

    auto numScsIp = meSCS.numIntPoints_;
    auto nodesPerFace = meSCS.nodesPerElement_;

    std::vector<double> ws_scalar(nodesPerFace);
    std::vector<double> ws_dualVolume(nodesPerFace);
    std::vector<double> ws_coords(2*nodesPerFace);
    std::vector<double> ws_areav(2*numScsIp);
    std::vector<double> ws_dqdx(2*numScsIp);
    const auto* ipNodeMap = meSCS.ipNodeMap();

    std::vector<double> ws_shape_function(nodesPerFace*numScsIp);
    meSCS.shape_fcn(ws_shape_function.data());

    for (size_t k = 0; k < length; ++k) {
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
      meSCS.determinant(1, ws_coords.data(), ws_areav.data(), &scs_error);

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
PromoteElementTest::dump_coords()
{
  stk::mesh::BucketVector const& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, metaData_->universal_part());
  for (const auto ib : elem_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const auto elem = b[k];
      stk::mesh::Entity const* node_rels = promoteElement_->begin_nodes_all(elem);

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
        else if (meshSpacing != dist) {
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
    std::cout <<
        "The test assumes that the mesh is uniform and "
        "will definitely fail the dual nodal volume check otherwise."
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

  bool testPassed = false;

  double tol = 1.0e-8; //P=5 is off on this test base 6.07e-10

  std::vector<double> zeroVec(nDim_,0.0);
  std::vector<double> ws_dqdx(nDim_);

  stk::mesh::Selector s_all_entities = metaData_->universal_part();
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
      bulkData_->get_buckets(stk::topology::ELEM_RANK, metaData_->universal_part());
  for (const auto* ib : elem_buckets ) {
    const stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      auto elem  = b[k];
      stk::mesh::Entity const* node_rels = promoteElement_->begin_nodes_all(elem);
      for (unsigned j = 0; j < promoteElement_->num_nodes(elem); ++j) {
        const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node_rels[j]);
        const auto num_elems = *stk::mesh::field_data(*sharedElems_, node_rels[j]);
        double exact = num_elems*exactDualNodalVolume[j];
        if (!is_near(dualNodalVolume,exact)) {
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

  const double scsDist = std::sqrt(3.0) / 3.0;
  const double corner = (1.0 - scsDist) * (meshSpacing * 0.5);
  const double center = 2.0 * scsDist * ( meshSpacing * 0.5);

  //varieties
  const double cornerVolume = corner * corner * corner;
  const double edgeVolume   = corner * corner * center;
  const double faceVolume   = corner * center * center;
  const double centerVolume = center * center * center;

  auto selector = stk::mesh::selectUnion(originalPartVector_);
  stk::mesh::BucketVector const& elem_buckets =
      bulkData_->get_buckets(stk::topology::ELEM_RANK, selector);
  for (const auto ib : elem_buckets ) {
    stk::mesh::Bucket & b = *ib ;
    const stk::mesh::Bucket::size_type length = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity const* node_rels = promoteElement_->begin_nodes_all(b, k);

      //corner nodes
      for (size_t j = 0; j < 8; ++j) {
        const stk::mesh::Entity node = node_rels[j];
        const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node);
        const auto num_elems = *stk::mesh::field_data(*sharedElems_,node);
        const double error = (num_elems*cornerVolume) - dualNodalVolume;
        if (std::abs(error) > floatingPointTolerance_) {
          std::cout << "Corner, Exact: " << (num_elems*cornerVolume)
              << " calculated: " << dualNodalVolume << std::endl;
          return false;
        }
      }

      //edge nodes
      for (size_t j = 8; j < 20; ++j) {
        const stk::mesh::Entity node = node_rels[j];
        const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node);
        const auto num_elems = *stk::mesh::field_data(*sharedElems_,node);
        const double error = (num_elems*edgeVolume) - dualNodalVolume;
        if (std::abs(error) > floatingPointTolerance_) {
          std::cout << "Edge, Exact: " << (num_elems*edgeVolume)
              << " calculated: " << dualNodalVolume << std::endl;
          return false;
        }
      }

      //center node
      const stk::mesh::Entity node = node_rels[20];
      const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node);
      const auto num_elems = *stk::mesh::field_data(*sharedElems_,node);
      if (num_elems != 1 ) {
        return false;
      }
      const double error = (num_elems*centerVolume) - dualNodalVolume;
      if (std::abs(error) > floatingPointTolerance_) {
        std::cout << "Center, Exact: " << (num_elems*centerVolume)
            << " calculated: " << dualNodalVolume << std::endl;
        return false;
      }
      //face nodes
      for (size_t j = 21; j < 27; ++j) {
        const stk::mesh::Entity node = node_rels[j];
        const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_,node);
        const auto num_elems = *stk::mesh::field_data(*sharedElems_,node);
        const double error = (num_elems*faceVolume) - dualNodalVolume;
        if (std::abs(error) > floatingPointTolerance_) {
          std::cout << "Face, Exact: " << (num_elems*faceVolume)
              << " calculated: " << dualNodalVolume << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}
//--------------------------------------------------------------------------
void
PromoteElementTest::register_fields()
{
  // extract blocks in the mesh with target names that are specified inline
  std::vector<std::string> targetNames;
  targetNames.push_back("block_1");
  if (nDim_ == 2) {
    targetNames.push_back("surface_1");
    targetNames.push_back("surface_2");
    targetNames.push_back("surface_3");
    targetNames.push_back("surface_4");
  }
  else {
    targetNames.push_back("surface_1");
    targetNames.push_back("surface_2");
    targetNames.push_back("surface_3");
    targetNames.push_back("surface_4");
    targetNames.push_back("surface_5");
    targetNames.push_back("surface_6");
  }

  std::vector<std::string> promotedNames;

  // save space for parts of the input mesh
  for (auto& targetName : targetNames) {
    std::string promotedName = targetName + "_promoted";

    stk::mesh::Part* targetPart = metaData_->get_part(targetName);
    stk::mesh::Part* promotedPart = &metaData_->declare_part(promotedName);

    // extract the parts
    originalPartVector_.push_back(targetPart);
    promotedPartVector_.push_back(promotedPart);

    // register nodal fields
    dualNodalVolume_ = &(metaData_->declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"));
    stk::mesh::put_field(*dualNodalVolume_, *targetPart);
    stk::mesh::put_field(*dualNodalVolume_, *promotedPart);

    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_,*targetPart, nDim_);
    stk::mesh::put_field(*coordinates_,*promotedPart, nDim_);

    dqdx_ = &(metaData_->declare_field<VectorFieldType>(
       stk::topology::NODE_RANK, "dqdx"));
     stk::mesh::put_field(*dqdx_,*targetPart, nDim_);
     stk::mesh::put_field(*dqdx_,*promotedPart, nDim_);

    // slightly helpful for checking dual nodal volume
    sharedElems_ = &(metaData_->declare_field<ScalarIntFieldType>(
      stk::topology::NODE_RANK, "elements_shared"));
    stk::mesh::put_field(*sharedElems_, *targetPart);
    stk::mesh::put_field(*sharedElems_, *promotedPart);

    q_ = &(metaData_->
        declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *targetPart);
    stk::mesh::put_field(*q_, *promotedPart);
   }
}
//--------------------------------------------------------------------------
void
PromoteElementTest::set_output_fields()
{
  resultsFileIndex_ =
      ioBroker_->create_output_mesh(coarseOutputName_, stk::io::WRITE_RESULTS );

  ioBroker_->add_field(resultsFileIndex_, *dualNodalVolume_, dualNodalVolume_->name());

  promoteIO_ = make_unique<PromotedElementIO>(
    *promoteElement_,
    *metaData_,
    *bulkData_,
    *coordinates_,
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
        q[k] = (std::cos(2.0*pi*coords[0+k*nDim_])+std::cos(2.0*pi*coords[1+k*nDim_]))*0.25;
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
  promoteIO_->write_database_data(currentTime_);
}
//--------------------------------------------------------------------------
bool PromoteElementTest::is_near(double approx, double exact)
{
  return (std::abs(approx-exact) < floatingPointTolerance_);
}
//--------------------------------------------------------------------------
bool PromoteElementTest::is_near(
  const std::vector<double>& approx,
  const std::vector<double>& exact)
{
  bool is_near = false;
  if (approx.size() == exact.size()) {
    for (unsigned j = 0; j < approx.size(); ++j) {
      if (std::abs(approx[j] - exact[j]) >= floatingPointTolerance_) {
        return false;
      }
      else {
        is_near = true;
      }
    }
  }
  else {
    return false;
  }
  return is_near;
}
//--------------------------------------------------------------------------
bool PromoteElementTest::is_near(
  const std::vector<double>& approx,
  const std::vector<double>& exact,
  double tolerance)
{
  bool is_near = false;
  if (approx.size() == exact.size()) {
    for (unsigned j = 0; j < approx.size(); ++j) {
      if (std::abs(approx[j] - exact[j]) >= tolerance) {
        return false;
      }
      else {
        is_near = true;
      }
    }
  }
  else {
    return false;
  }
  return is_near;
}

} // namespace naluUnit
}  // namespace sierra

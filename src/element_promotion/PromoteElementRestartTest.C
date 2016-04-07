/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromoteElementRestartTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/MasterElementHO.h>
#include <element_promotion/MasterElement.h>

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
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/environment/ReportHandler.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

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
// PromoteElementRestartTests - a set of tests for element promotion restart
// Read mesh in, check whether the dual nodal volume field is the exact
// dual nodal volume for the expected mesh
// TODO(rcknaus): separate out the promote element tests so that they don't
// to be copied here
//==========================================================================
PromoteElementRestartTest::PromoteElementRestartTest(std::string restartName, std::string outputFileName)
  :  restartFileName_(std::move(restartName)),
     outputFileName_(std::move(outputFileName)),
     restartFileIndex_(1),
     resultsFileIndex_(2),
     defaultFloatingPointTolerance_(1.0e-12)
{
}
//--------------------------------------------------------------------------
PromoteElementRestartTest::~PromoteElementRestartTest() = default;
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::execute()
{
  read_restart_mesh();

  auto meshParts = metaData_->get_mesh_parts();
  baseElemParts_ = base_elem_parts(meshParts);
  superElemParts_ = only_super_elem_parts(meshParts);

  //Require (for now) that all base element parts were promoted
  ThrowRequire(super_elem_part_vector(baseElemParts_) == superElemParts_
            && !superElemParts_.empty());

  const unsigned numNodes = superElemParts_.at(0)->topology().num_nodes();
  unsigned polyOrder = (metaData_->spatial_dimension() == 2) ?
      std::sqrt(static_cast<double>(numNodes+1))-1 : std::cbrt(static_cast<double>(numNodes+1))-1;
  ThrowRequire(std::pow(polyOrder+1, metaData_->spatial_dimension()) == numNodes);

  std::string elemName;
  if(metaData_->spatial_dimension() == 2) {
    unsigned nodes = (polyOrder+1)*(polyOrder+1);
    elemName = "Quad" + std::to_string(nodes);
  }
  else {
    unsigned nodes = (polyOrder+1)*(polyOrder+1)*(polyOrder+1);
    elemName = "Hex" + std::to_string(nodes);
  }

  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;
  NaluEnv::self().naluOutputP0() << "Restarting with a '" << elemName
                                 << "' Element"
                                 <<   std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------"  << std::endl;

  elem_ = ElementDescription::create(metaData_->spatial_dimension(), polyOrder);
  meSCS_ = create_master_subcontrol_surface_element(*elem_);
  meBC_  = create_master_boundary_element(*elem_);

  promoteElement_ = make_unique<PromoteElement>(*elem_);

  promoteElement_->populate_boundary_connectivity_map_using_super_elems(
    *bulkData_,
    meshParts
  );

  read_input_fields();
  set_output_fields();

  compute_projected_nodal_gradient();

  output_results();
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::read_restart_mesh()
{
  stk::ParallelMachine pm = NaluEnv::self().parallel_comm();

  //mesh setup
  metaData_ = make_unique<stk::mesh::MetaData>();
  bulkData_ = make_unique<stk::mesh::BulkData>(*metaData_, pm, stk::mesh::BulkData::NO_AUTO_AURA);
  ioBroker_ = make_unique<stk::io::StkMeshIoBroker>(pm);
  ioBroker_->set_bulk_data(*bulkData_);

  // deal with input mesh
  ioBroker_->add_mesh_database(restartFileName_, stk::io::READ_RESTART);
  ioBroker_->create_input_mesh();
  nDim_ = metaData_->spatial_dimension();

  register_fields();
  ioBroker_->add_input_field({*dualNodalVolume_, dualNodalVolume_->name()});
  ioBroker_->add_input_field({*sharedElems_, sharedElems_->name()});
  ioBroker_->add_input_field({*q_, q_->name()});

  ioBroker_->populate_bulk_data();
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::register_fields()
{
  // register fields on all mesh parts
  for (const auto* ipart : metaData_->get_mesh_parts()) {

    dualNodalVolume_ = &(metaData_->declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"));
    stk::mesh::put_field(*dualNodalVolume_, *ipart);

    sharedElems_ = &(metaData_->declare_field<ScalarIntFieldType>(
      stk::topology::NODE_RANK, "elements_shared"));
    stk::mesh::put_field(*sharedElems_, *ipart);

    coordinates_ = &(metaData_->declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates"));
    stk::mesh::put_field(*coordinates_,*ipart, nDim_);

    q_ = &(metaData_->
        declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar"));
    stk::mesh::put_field(*q_, *ipart);

    dqdx_ = &(metaData_->declare_field<VectorFieldType>(
       stk::topology::NODE_RANK, "dqdx"));
     stk::mesh::put_field(*dqdx_,*ipart, nDim_);
  }
}
void
PromoteElementRestartTest::read_input_fields()
{
  std::vector<stk::io::MeshField> missingFields;
  ioBroker_->read_defined_input_fields(0.0, &missingFields);
  ThrowRequire(missingFields.empty());
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::set_output_fields()
{
  promoteIO_ = make_unique<PromotedElementIO>(
    *elem_,
    *metaData_,
    *bulkData_,
    baseElemParts_,
    outputFileName_
  );

  promoteIO_->add_fields({dualNodalVolume_, q_, dqdx_});
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::output_results()
{
 promoteIO_->write_database_data(0.0);
 output_result("Restart PNG", check_projected_nodal_gradient());
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::compute_projected_nodal_gradient()
{
  auto interiorSelector = stk::mesh::selectUnion(superElemParts_)
                & metaData_->locally_owned_part();

  auto boundarySelector = stk::mesh::selectUnion(metaData_->get_mesh_parts())
                & metaData_->locally_owned_part();

  stk::mesh::field_fill(0.0,*dqdx_, interiorSelector);

  compute_projected_nodal_gradient_interior(interiorSelector);
  compute_projected_nodal_gradient_boundary(boundarySelector);

  if (bulkData_->parallel_size() > 1) {
    stk::mesh::parallel_sum(*bulkData_, {dqdx_});
  }
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::compute_projected_nodal_gradient_interior(
  stk::mesh::Selector& selector)
{
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
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::compute_projected_nodal_gradient_boundary(
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
bool
PromoteElementRestartTest::check_projected_nodal_gradient()
{
  // test assumes that the scalar q is linear
  // will fail otherwise

  bool testPassed = false;

  double tol = 1.0e-8; //P=5 is off on this test by 6.07e-10

  std::vector<double> oneVec(nDim_,1.0);
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

      if (is_near(ws_dqdx, oneVec, tol)) {
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
std::unique_ptr<MasterElement>
PromoteElementRestartTest::create_master_subcontrol_surface_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderQuad2DSCS>(elem);
  }
  return make_unique<HigherOrderHexSCS>(elem);
}
//--------------------------------------------------------------------------
std::unique_ptr<MasterElement>
PromoteElementRestartTest::create_master_boundary_element(const ElementDescription& elem)
{
  if (elem.dimension == 2) {
    return make_unique<HigherOrderEdge2DSCS>(elem);
  }
  return make_unique<HigherOrderQuad3DSCS>(elem);
}

} // namespace naluUnit
}  // namespace sierra

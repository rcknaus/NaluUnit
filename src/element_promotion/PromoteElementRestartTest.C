/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromoteElementRestartTest.h>

#include <NaluEnv.h>
#include <element_promotion/ElementDescription.h>
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/PromotedPartHelper.h>
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
// PromoteElementRestartTests - a set of tests for element promotion restart
// Read mesh in, check whether the dual nodal volume field is the exact
// dual nodal volume for the expected mesh
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

  read_input_fields();
  set_output_fields();
  output_results();

  output_result("Restart DNV", check_dual_nodal_volume());
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

  promoteIO_->add_fields({dualNodalVolume_});
}
//--------------------------------------------------------------------------
void
PromoteElementRestartTest::output_results()
{
 promoteIO_->write_database_data(0.0);
}
//--------------------------------------------------------------------------
bool
PromoteElementRestartTest::check_dual_nodal_volume()
{
  if (nDim_ == 2) {
    return check_dual_nodal_volume_quad();
  }
  return check_dual_nodal_volume_hex();
}
//--------------------------------------------------------------------------
bool
PromoteElementRestartTest::check_dual_nodal_volume_quad()
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
      bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superElemParts_));
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
PromoteElementRestartTest::check_dual_nodal_volume_hex()
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
        bulkData_->get_buckets(stk::topology::ELEM_RANK, stk::mesh::selectUnion(superElemParts_));
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
double
PromoteElementRestartTest::determine_mesh_spacing()
{
  double meshSpacing = std::numeric_limits<double>::lowest();
  bool isUniform = true;
  auto selector = stk::mesh::selectUnion(baseElemParts_);
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


} // namespace naluUnit
}  // namespace sierra

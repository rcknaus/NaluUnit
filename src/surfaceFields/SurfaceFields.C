/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <surfaceFields/SurfaceFields.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/SkinMesh.hpp>

// stk_io
#include <stk_io/StkMeshIoBroker.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// SurfaceFields - what is worng with field data retrieval?
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SurfaceFields::SurfaceFields()
  : activateAura_(false),
    currentTime_(0.0),
    resultsFileIndex_(1),
    nDim_(2),
    metaData_(NULL),
    bulkData_(NULL),
    ioBroker_(NULL),
    doSubset_(true)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SurfaceFields::~SurfaceFields()
{
  delete bulkData_;
  delete metaData_;
  delete ioBroker_;
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void 
SurfaceFields::execute() 
{
  NaluEnv::self().naluOutputP0() << "Welcome to the SurfaceFields unit test" << std::endl;

   stk::ParallelMachine pm = NaluEnv::self().parallel_comm();
  
   // news for mesh constructs
   metaData_ = new stk::mesh::MetaData();
   bulkData_ 
     = new stk::mesh::BulkData(*metaData_, pm, activateAura_ ? stk::mesh::BulkData::AUTO_AURA : stk::mesh::BulkData::NO_AUTO_AURA);
   ioBroker_ = new stk::io::StkMeshIoBroker( pm );
   ioBroker_->set_bulk_data(*bulkData_);

   // deal with input mesh
   ioBroker_->add_mesh_database( "waterChannel_mks.g", stk::io::READ_MESH );
   ioBroker_->create_input_mesh();

   register_fields();

   // populate bulk data
   ioBroker_->populate_bulk_data();

   // deal with output mesh
   set_output_fields();

   // safe to set nDim
   nDim_ = metaData_->spatial_dimension();
 
   // initialize nodal fields
   initialize_fields();   

   // check for null
   check_for_null();

   // output results
   output_results();
}


//--------------------------------------------------------------------------
//-------- register_fields -------------------------------------------------
//--------------------------------------------------------------------------
void 
SurfaceFields::register_fields()
{

  // declare the part
  normalHeatFlux_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "normal_heat_flux"));

  // now extract blocks in the mesh with target names for "put_field"
  std::vector<std::string> targetNames;
  targetNames.push_back("surface_3");
  targetNames.push_back("surface_4");
  targetNames.push_back("surface_5");
  
  // save space for parts of the input mesh
  for ( size_t itarget = 0; itarget < targetNames.size(); ++itarget ) {

    // extract the part
    stk::mesh::Part *targetPart = metaData_->get_part(targetNames[itarget]);

    if ( doSubset_ ) {
      const std::vector<stk::mesh::Part*> & mesh_parts = targetPart->subsets();
      for( std::vector<stk::mesh::Part*>::const_iterator i = mesh_parts.begin();
           i != mesh_parts.end(); ++i ) {
        stk::mesh::Part * const subsetPart = *i ;

        // push
        surfacePartVector_.push_back(subsetPart);
      
        NaluEnv::self().naluOutputP0() << "Subsetting on part: " << subsetPart->name() << std::endl;

        // put nodal fields
        stk::mesh::put_field(*normalHeatFlux_, *targetPart);    
      }
    }
    else {
      // push
      surfacePartVector_.push_back(targetPart);
      
      // put nodal fields
      stk::mesh::put_field(*normalHeatFlux_, *targetPart);    
    }
  }
}
  
//--------------------------------------------------------------------------
//-------- set_output_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceFields::set_output_fields()
{  
  resultsFileIndex_ = ioBroker_->create_output_mesh( "surfaceFieldsOutput.e", stk::io::WRITE_RESULTS );
  ioBroker_->add_field(resultsFileIndex_, *normalHeatFlux_, normalHeatFlux_->name());
}
  
//--------------------------------------------------------------------------
//-------- initialize_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceFields::initialize_fields()
{
  for ( size_t sp = 0; sp < surfacePartVector_.size(); ++sp) {

    // extract the part and name
    stk::mesh::Part *thePart = surfacePartVector_[sp];
    std::string partName = thePart->name();

    // give the value as a function of the part name
    double partValue = 0.0;
    if ( partName == "surface_3" )
      partValue = 3.0;
    else if ( partName == "surface_4" )
      partValue = 4.0;
    else
      partValue = 5.0;

    stk::mesh::Selector s_all_entities
      = (metaData_->locally_owned_part() | metaData_->globally_shared_part())
      & stk::mesh::Selector(*thePart);

    stk::mesh::BucketVector const& node_buckets = bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
    for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
          ib != node_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      const stk::mesh::Bucket::size_type length   = b.size();
      double * normalHeatFlux = stk::mesh::field_data(*normalHeatFlux_, b);
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        normalHeatFlux[k] = partValue;
      }
    }  
  }
}

//--------------------------------------------------------------------------
//-------- check_for_null --------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceFields::check_for_null()
{
  const bool pushSurface3 = false;
  const bool pushSurface4 = false;
  const bool pushSurface5 = true;
  std::vector<stk::mesh::Part*> testVec;

  if ( pushSurface3 ) {
    stk::mesh::Part *targetPart = metaData_->get_part("surface_3");
    testVec.push_back(targetPart);
    std::cout << "Push back on: " << targetPart->name() << std::endl;
  }

  if ( pushSurface4 ) {
    stk::mesh::Part *targetPart = metaData_->get_part("surface_4");
    testVec.push_back(targetPart);
    std::cout << "Push back on: " << targetPart->name() << std::endl;
  }

  if ( pushSurface5 ) {
    stk::mesh::Part *targetPart = metaData_->get_part("surface_5");
    testVec.push_back(targetPart);
    std::cout << "Push back on: " << targetPart->name() << std::endl;
  }

  size_t numNull = 0;

  stk::mesh::Selector s_all_nodes_now
    = (metaData_->locally_owned_part() | metaData_->globally_shared_part())
    & selectUnion(testVec);
  
  stk::mesh::BucketVector const& node_buckets_now =
    bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_nodes_now );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets_now.begin() ;
        ib != node_buckets_now.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      stk::mesh::Entity node = b[k];
      const double * primitive = (double*)stk::mesh::field_data(*normalHeatFlux_, node);
      if ( NULL == primitive ) {
        numNull++;
      }
    }
  }
  
  size_t g_numNull = 0;
  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_min(comm, &numNull, &g_numNull, 1);

  if ( g_numNull > 0 ) {
    NaluEnv::self().naluOutputP0() << "SurfaceFields::FAIL: " << g_numNull << std::endl;
  }
  else {
    NaluEnv::self().naluOutputP0() << "SurfaceFields::PASS: " << g_numNull << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- output_results -----------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceFields::output_results()
{
  ioBroker_->process_output_request(resultsFileIndex_, currentTime_);
}

} // namespace naluUnit
} // namespace Sierra

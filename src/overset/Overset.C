/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <overset/Overset.h>
#include <overset/OversetInfo.h>
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

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// stk_io
#include <stk_io/StkMeshIoBroker.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// Overset - unit test for overset
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Overset::Overset()
  : activateAura_(false),
    singleOversetBox_(true),
    reductionFactor_(0.10),
    searchMethod_(stk::search::BOOST_RTREE),
    currentTime_(0.0),
    resultsFileIndex_(1),
    nDim_(2),
    metaData_(NULL),
    bulkData_(NULL),
    ioBroker_(NULL),
    nodeBackgroundMesh_(NULL),
    elemBackgroundMesh_(NULL),
    nodeOversetMesh_(NULL),
    elemOversetMesh_(NULL),
    nodeIntersectedMesh_(NULL),
    elemIntersectedMesh_(NULL),
    coordinates_(NULL),
    inActivePart_(NULL),
    backgroundSurfacePart_(NULL)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Overset::~Overset()
{
  delete bulkData_;
  delete metaData_;
  delete ioBroker_;

  // delete the overset info objects
  std::vector<OversetInfo*>::iterator ii;
  for( ii=oversetInfoVec_.begin(); ii!=oversetInfoVec_.end(); ++ii )
    delete (*ii);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void 
Overset::execute() 
{
  NaluEnv::self().naluOutputP0() << "Welcome to the Overset unit test";

   stk::ParallelMachine pm = NaluEnv::self().parallel_comm();
  
   // news for mesh constructs
   metaData_ = new stk::mesh::MetaData();
   bulkData_ = new stk::mesh::BulkData(*metaData_, pm, activateAura_ ? stk::mesh::BulkData::AUTO_AURA : stk::mesh::BulkData::NO_AUTO_AURA);
   ioBroker_ = new stk::io::StkMeshIoBroker( pm );
   ioBroker_->set_bulk_data(*bulkData_);

   // deal with input mesh
   ioBroker_->add_mesh_database( "oversetMeshAligned.g", stk::io::READ_MESH );
   ioBroker_->create_input_mesh();

   register_fields();

   // create the part that represents the intersected elements/nodes; 
   declare_inactive_part();

   // create the part for the surface of the intersected elements/nodes
   declare_background_surface_part();

   // populate bulk data
   ioBroker_->populate_bulk_data();

   // deal with output mesh
   set_output_fields();

   // safe to set nDim
   nDim_ = metaData_->spatial_dimension();
 
   // extract coordinates
   coordinates_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
   
   // initialize nodal fields; define selector (locally owned and ghosted)
   initialize_fields();
   
   // define overset bounding box
   define_overset_bounding_box();

   // define background bounding box
   define_background_bounding_box();

   // perform the coarse search
   coarse_search();

   // create a part that holds the intersected elements that should be inactive
   create_inactive_part();

   // find the exposed surfaces and place in a part
   create_exposed_surface_on_inactive_part();

   // populate fringePointSurfaceVec_
   populate_exposed_surface_fringe_part_vec();

   // define OversetInfo object for each node on the exposed parts
   create_overset_info_vec();

   // search for points in elements;; isInElement
   fringe_point_search();

   // set the element 
   set_data_on_inactive_part();
   
   // set data on overset
   set_data_on_overset_part();

   // output results
   output_results();
}

//--------------------------------------------------------------------------
//-------- declare_inactive_part -------------------------------------------
//--------------------------------------------------------------------------
void
Overset::declare_inactive_part()
{
  // not sure where this needs to be in order for the blcok to show up in the output file?
  std::string partName = "block_3";
  inActivePart_ =  &metaData_->declare_part(partName, stk::topology::ELEMENT_RANK);
}

//--------------------------------------------------------------------------
//-------- declare_background_surface_part ---------------------------------
//--------------------------------------------------------------------------
void
Overset::declare_background_surface_part()
{
  // not sure where this needs to be in order for the blcok to show up in the output file?
  std::string partName = "surface_101";
  backgroundSurfacePart_ =  &metaData_->declare_part(partName, metaData_->side_rank());
}

//--------------------------------------------------------------------------
//-------- register_fields -------------------------------------------------
//--------------------------------------------------------------------------
void 
Overset::register_fields()
{
  // extract blocks in the mesh with target names that are specified inline
  std::vector<std::string> targetNames;
  targetNames.push_back("block_1");
  targetNames.push_back("block_2");
  
  // save space for parts of the input mesh
  for ( size_t itarget = 0; itarget < targetNames.size(); ++itarget ) {
    
    // extract the part
    stk::mesh::Part *targetPart = metaData_->get_part(targetNames[itarget]);
    
    // push back the part
    volumePartVector_.push_back(targetPart);
    
    // register nodal fields
    nodeBackgroundMesh_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_back_ground_mesh"));
    elemBackgroundMesh_ = &(metaData_->declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "elem_background_mesh"));
    
    nodeOversetMesh_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_overset_mesh"));
    elemOversetMesh_ = &(metaData_->declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "elem_overset_mesh"));
    
    nodeIntersectedMesh_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_intersected_mesh"));
    elemIntersectedMesh_ = &(metaData_->declare_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "elem_intersected_mesh"));
    
    // put them on the part
    const int sizeOfElemField = 1;
    stk::mesh::put_field(*nodeBackgroundMesh_, *targetPart);
    stk::mesh::put_field(*elemBackgroundMesh_, *targetPart, sizeOfElemField);
    
    stk::mesh::put_field(*nodeOversetMesh_, *targetPart);
    stk::mesh::put_field(*elemOversetMesh_, *targetPart, sizeOfElemField);
    
    stk::mesh::put_field(*nodeIntersectedMesh_, *targetPart);
    stk::mesh::put_field(*elemIntersectedMesh_, *targetPart, sizeOfElemField);
  }
}
  
//--------------------------------------------------------------------------
//-------- set_output_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
Overset::set_output_fields()
{  
  resultsFileIndex_ = ioBroker_->create_output_mesh( "oversetOutput.e", stk::io::WRITE_RESULTS );
  ioBroker_->add_field(resultsFileIndex_, *nodeBackgroundMesh_, nodeBackgroundMesh_->name());
  ioBroker_->add_field(resultsFileIndex_, *elemBackgroundMesh_, elemBackgroundMesh_->name());
  ioBroker_->add_field(resultsFileIndex_, *nodeOversetMesh_, nodeOversetMesh_->name());
  ioBroker_->add_field(resultsFileIndex_, *elemOversetMesh_, elemOversetMesh_->name());
  ioBroker_->add_field(resultsFileIndex_, *nodeIntersectedMesh_, nodeIntersectedMesh_->name());
  ioBroker_->add_field(resultsFileIndex_, *elemIntersectedMesh_, elemIntersectedMesh_->name());
}
  
//--------------------------------------------------------------------------
//-------- initialize_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
Overset::initialize_fields()
{
  stk::mesh::Selector s_all_entities = stk::mesh::selectUnion(volumePartVector_);
  
  stk::mesh::BucketVector const& node_buckets = bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * nodeBackgroundMesh = stk::mesh::field_data(*nodeBackgroundMesh_, b);
    double * nodeOversetMesh = stk::mesh::field_data(*nodeOversetMesh_, b);
    double * nodeIntersectedMesh = stk::mesh::field_data(*nodeIntersectedMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      nodeBackgroundMesh[k] = 1.0;
      nodeOversetMesh[k] = 2.0;
      nodeIntersectedMesh[k] = 0.0;
    }
  }
  
  // initialize element fields; use selector from above 
  stk::mesh::BucketVector const& elem_buckets = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_all_entities );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin() ;
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * elemBackgroundMesh = stk::mesh::field_data(*elemBackgroundMesh_, b);
    double * elemOversetMesh= stk::mesh::field_data(*elemOversetMesh_, b);
    double * elemIntersectedMesh = stk::mesh::field_data(*elemIntersectedMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      elemBackgroundMesh[k] = 1.0;
      elemOversetMesh[k] = 2.0;
      elemIntersectedMesh[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- define_overset_bounding_box -------------------------------------
//--------------------------------------------------------------------------
void
Overset::define_overset_bounding_box()
{
  // obtained via block_2 max/min coords
  if ( singleOversetBox_ ) {
    std::vector<double> minOversetCorner(nDim_);
    std::vector<double> maxOversetCorner(nDim_);
    
    // initialize max and min
    for (int j = 0; j < nDim_; ++j ) {
      minOversetCorner[j] = +1.0e16;
      maxOversetCorner[j] = -1.0e16;
    }
    
    // use locally owned elemetsjust for the sake of tutorial
    stk::mesh::Selector s_locally_owned_union_overset = metaData_->locally_owned_part()
      &stk::mesh::Selector(*volumePartVector_[1]);
    
    stk::mesh::BucketVector const& locally_owned_elem_buckets =
      bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_overset );
    
    for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets.begin();
          ib != locally_owned_elem_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib;
      
      const stk::mesh::Bucket::size_type length   = b.size();
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        // get element
        stk::mesh::Entity element = b[k];
        
        // extract elem_node_relations
        stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
        const int num_nodes = bulkData_->num_nodes(element);
        
        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = elem_node_rels[ni];
          
          // pointers to real data
          const double * coords = stk::mesh::field_data(*coordinates_, node );
          
          // check max/min
          for ( int j = 0; j < nDim_; ++j ) {
            minOversetCorner[j] = std::min(minOversetCorner[j], coords[j]);
            maxOversetCorner[j] = std::max(maxOversetCorner[j], coords[j]);
          }
        }
      }
    }
    
    // parallel reduce max/min
    std::vector<double> g_minOversetCorner(nDim_);
    std::vector<double> g_maxOversetCorner(nDim_);
    
    stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
    stk::all_reduce_min(comm, &minOversetCorner[0], &g_minOversetCorner[0], nDim_);
    stk::all_reduce_max(comm, &maxOversetCorner[0], &g_maxOversetCorner[0], nDim_);
    
    // copy to the point; with reduction below, be very deliberate since these are cheap loops
    Point minOverset;
    Point maxOverset;

    // determine translation distance to provide minimum origin at (0,0,0)
    std::vector<double> translateDistance(nDim_);
    for ( int i = 0; i < nDim_; ++i ) {
      translateDistance[i] = -g_minOversetCorner[i];
    }

    // translate
    for ( int i = 0; i < nDim_; ++i ) {
      g_minOversetCorner[i] += translateDistance[i];
      g_maxOversetCorner[i] += translateDistance[i];
    }

    // reduce the translated box
    for ( int i = 0; i < nDim_; ++i ) {
      const double distance = (g_maxOversetCorner[i] - g_minOversetCorner[i])*reductionFactor_;
      g_minOversetCorner[i] += distance;
      g_maxOversetCorner[i] -= distance;
    }
    
    // translate back to original
    for ( int i = 0; i < nDim_; ++i ) {
      g_minOversetCorner[i] -= translateDistance[i];
      g_maxOversetCorner[i] -= translateDistance[i];
    }
    
    // inform the user and copy into points
    NaluEnv::self().naluOutputP0() << "Min/Max coords for overset bounding box" << std::endl;
    for ( int i = 0; i < nDim_; ++i ) {      
      minOverset[i] = g_minOversetCorner[i];
      maxOverset[i] = g_maxOversetCorner[i];
      NaluEnv::self().naluOutputP0() << "component: " << i << " " << minOverset[i] << " " << maxOverset[i] << std::endl;
    }
    
    // set up the processor info for this bounding box; attach it to local rank with unique id
    const size_t overSetBoundingBoxIdent = 0;
    const int parallelRankForBoundingBox = NaluEnv::self().parallel_rank();
    stk::search::IdentProc<uint64_t,int> theIdent(overSetBoundingBoxIdent, parallelRankForBoundingBox);
    
    // bounding box for all of the overset mesh
    boundingElementBox oversetBox(Box(minOverset,maxOverset), theIdent);
    boundingElementOversetBoxVec_.push_back(oversetBox);
  }
  else {

    cut_surface();

    // setup data structures for search
    Point minOversetCorner, maxOversetCorner;
    
    stk::mesh::Selector s_locally_owned_union_over = metaData_->locally_owned_part()
      &stk::mesh::Selector(*volumePartVector_[1]);
    
    stk::mesh::BucketVector const& locally_owned_elem_buckets_over =
      bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_over );
    
    for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets_over.begin();
          ib != locally_owned_elem_buckets_over.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib;
      
      const stk::mesh::Bucket::size_type length   = b.size();
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        // get element
        stk::mesh::Entity element = b[k];
        
        // initialize max and min
        for (int j = 0; j < nDim_; ++j ) {
          minOversetCorner[j] = +1.0e16;
          maxOversetCorner[j] = -1.0e16;
        }
        
        // extract elem_node_relations
        stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
        const int num_nodes = bulkData_->num_nodes(element);
        
        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = elem_node_rels[ni];
          
          // pointers to real data
          const double * coords = stk::mesh::field_data(*coordinates_, node );
          
          // check max/min
          for ( int j = 0; j < nDim_; ++j ) {
            minOversetCorner[j] = std::min(minOversetCorner[j], coords[j]);
            maxOversetCorner[j] = std::max(maxOversetCorner[j], coords[j]);
          }
        }
        
        // setup ident
        stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(element), NaluEnv::self().parallel_rank());
        
        // create the bounding point box and push back
        boundingElementBox theBox(Box(minOversetCorner,maxOversetCorner), theIdent);
        boundingElementOversetBoxVec_.push_back(theBox);
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- cut_surface -----------------------------------------------------
//--------------------------------------------------------------------------
void
Overset::cut_surface()
{
  std::vector<double> minOversetCorner(nDim_);
  std::vector<double> maxOversetCorner(nDim_);

  // initialize max and min
  for (int j = 0; j < nDim_; ++j ) {
    minOversetCorner[j] = +1.0e16;
    maxOversetCorner[j] = -1.0e16;
  }
  
  // use locally owned faces; first need to extract the part for surface_5
  stk::mesh::Part *targetPart = metaData_->get_part("surface_5");
  if ( NULL == targetPart ) {
    NaluEnv::self().naluOutputP0() << "Sorry, no part name found by the name surface_5"  << std::endl;
  }

  stk::mesh::Selector s_locally_owned_union_overset = metaData_->locally_owned_part()
    &stk::mesh::Selector(*targetPart);
  
  stk::mesh::BucketVector const& locally_owned_face_buckets =
    bulkData_->get_buckets( metaData_->side_rank(), s_locally_owned_union_overset );
  
  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_face_buckets.begin();
        ib != locally_owned_face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get face
      stk::mesh::Entity face = b[k];
      
      // extract elem_node_relations
      stk::mesh::Entity const* face_node_rels = bulkData_->begin_nodes(face);
      const int num_nodes = bulkData_->num_nodes(face);
      
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        
        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates_, node );
        
        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minOversetCorner[j] = std::min(minOversetCorner[j], coords[j]);
          maxOversetCorner[j] = std::max(maxOversetCorner[j], coords[j]);
        }
      }
    }
  }
  
  // parallel reduce max/min
  std::vector<double> g_minOversetCorner(nDim_);
  std::vector<double> g_maxOversetCorner(nDim_);
  
  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_min(comm, &minOversetCorner[0], &g_minOversetCorner[0], nDim_);
  stk::all_reduce_max(comm, &maxOversetCorner[0], &g_maxOversetCorner[0], nDim_);
  
  // copy to the point
  Point minOverset;
  Point maxOverset;
  
  NaluEnv::self().naluOutputP0() << "Min/Max coords for overset surface bounding box" << std::endl;
  for ( int i = 0; i < nDim_; ++i ) {
    minOverset[i] = g_minOversetCorner[i];
    maxOverset[i] = g_maxOversetCorner[i];
    NaluEnv::self().naluOutputP0() << "componenet: " << i << " " << minOverset[i] << " " << maxOverset[i] << std::endl;
  }
  
  // set up the processor infor for this bounding box; attach it to rank 0 with id 0
  const size_t overSetBoundingBoxIdent = 0;
  //const int parallelRankForBoundingBox = 0;
  stk::search::IdentProc<uint64_t,int> theIdent(overSetBoundingBoxIdent, 0);
  
  // bounding box for all of the overset mesh
  boundingElementBox oversetBox(Box(minOverset,maxOverset), theIdent);
  boundingElementOversetBoxVec_.push_back(oversetBox);
}

//--------------------------------------------------------------------------
//-------- define_background_bounding_box ----------------------------------
//--------------------------------------------------------------------------
void
Overset::define_background_bounding_box()
{
  // obtained via block_1 max/min coords
  
  // setup data structures for search
  Point minBackgroundCorner, maxBackgroundCorner;
  
  stk::mesh::Selector s_locally_owned_union_back = metaData_->locally_owned_part()
    &stk::mesh::Selector(*volumePartVector_[0]);
  
  stk::mesh::BucketVector const& locally_owned_elem_buckets_back =
    bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union_back );
  
  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_elem_buckets_back.begin();
        ib != locally_owned_elem_buckets_back.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get element
      stk::mesh::Entity element = b[k];
      
      // initialize max and min
      for (int j = 0; j < nDim_; ++j ) {
        minBackgroundCorner[j] = +1.0e16;
        maxBackgroundCorner[j] = -1.0e16;
      }
      
      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
      const int num_nodes = bulkData_->num_nodes(element);
      
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        
        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates_, node );
        
        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minBackgroundCorner[j] = std::min(minBackgroundCorner[j], coords[j]);
          maxBackgroundCorner[j] = std::max(maxBackgroundCorner[j], coords[j]);
        }
      }
      
      // setup ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(element), NaluEnv::self().parallel_rank());
      
      searchElementMap_[bulkData_->identifier(element)] = element;
      
      // create the bounding point box and push back
      boundingElementBox theBox(Box(minBackgroundCorner,maxBackgroundCorner), theIdent);
      boundingElementBackgroundBoxVec_.push_back(theBox);
    }
  }
}

//--------------------------------------------------------------------------
//-------- coarse_search ---------------------------------------------------
//--------------------------------------------------------------------------
void
Overset::coarse_search()
{
  stk::search::coarse_search(boundingElementOversetBoxVec_, boundingElementBackgroundBoxVec_, searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair_);
  
  // iterate search key; extract found elements and push to vector
  std::vector<std::pair<theKey, theKey> >::const_iterator ii;
  for( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {
    
    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned box_proc = ii->second.proc();
    
    // if this box in on-rank, extract the element; otherwise, do not worry about it
    if ( box_proc == theRank ) {
      // find the element
      std::map<uint64_t, stk::mesh::Entity>::iterator iterEM;
      iterEM=searchElementMap_.find(theBox);
      if ( iterEM == searchElementMap_.end() )
        throw std::runtime_error("No entry in searchElementMap found");
      stk::mesh::Entity theElemMeshObj = iterEM->second;
      
      intersectedElementVec_.push_back(theElemMeshObj);
    }
  }
}

//--------------------------------------------------------------------------
//-------- create_inactive_part --------------------------------------------
//--------------------------------------------------------------------------
void
Overset::create_inactive_part()
{
  // push all elements intersected to a new part
  bulkData_->modification_begin();

  stk::mesh::PartVector theNewPartVector;
  theNewPartVector.push_back(inActivePart_);
  for ( size_t k = 0; k < intersectedElementVec_.size(); ++k ) {
    stk::mesh::Entity theElement = intersectedElementVec_[k];
    // add to block_3
    bulkData_->change_entity_parts( theElement, theNewPartVector);
  }
  bulkData_->modification_end();

  // try to remove these elements from block_1... fails in a later nodal loop..
  const bool removeOld = false;
  if ( removeOld ) {
    bulkData_->modification_begin();
    stk::mesh::PartVector theOldPartVector;
    stk::mesh::PartVector theEmptyPartVector;
    theOldPartVector.push_back(volumePartVector_[0]);
    for ( size_t k = 0; k < intersectedElementVec_.size(); ++k ) {
      stk::mesh::Entity theElement = intersectedElementVec_[k];
      // remove from block_1
      bulkData_->change_entity_parts( theElement, theEmptyPartVector, theOldPartVector);
    }
    bulkData_->modification_end();  
  }

}
//--------------------------------------------------------------------------
//-------- create_exposed_surface_on_inactive_part -------------------------
//--------------------------------------------------------------------------
void
Overset::create_exposed_surface_on_inactive_part()
{
  // skin the inactive part to obtain all exposed surface
  stk::mesh::Selector s_inactive = stk::mesh::Selector(*inActivePart_);
  stk::mesh::PartVector partToSkinVec;
  stk::mesh::PartVector partToPopulateVec;
  partToSkinVec.push_back(inActivePart_); // block_3
  partToPopulateVec.push_back(backgroundSurfacePart_); // surface_101
  stk::mesh::skin_mesh(*bulkData_, s_inactive, partToPopulateVec, &s_inactive);

  // now output for debug purposes
  stk::mesh::Selector s_locally_owned = metaData_->locally_owned_part()
    &stk::mesh::Selector(*backgroundSurfacePart_);

  stk::mesh::BucketVector const& locally_owned_node_bucket =
    bulkData_->get_buckets( stk::topology::NODE_RANK, s_locally_owned );

  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_node_bucket.begin();
        ib != locally_owned_node_bucket.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get node
      stk::mesh::Entity node = b[k];
      
      // output identifier
      NaluEnv::self().naluOutputP0() << "nodes on surface id " << bulkData_->identifier(node) << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- populate_exposed_surface_fringe_part_vec ------------------------
//--------------------------------------------------------------------------
void
Overset::populate_exposed_surface_fringe_part_vec()
{
  // surface_6 and surface_101
  std::vector<std::string> targetNames;
  targetNames.push_back("surface_6");
  targetNames.push_back("surface_101");

  for ( size_t itarget = 0; itarget < targetNames.size(); ++itarget ) {
    
    // extract the part
    stk::mesh::Part *targetPart = metaData_->get_part(targetNames[itarget]);
    
    // push back the part
    fringePointSurfaceVec_.push_back(targetPart);
  }
}

//--------------------------------------------------------------------------
//-------- create_overset_info_vec -----------------------------------------
//--------------------------------------------------------------------------
void
Overset::create_overset_info_vec()
{
  Point localNodalCoords;

  stk::mesh::Selector s_locally_owned = metaData_->locally_owned_part()
    &stk::mesh::selectUnion(fringePointSurfaceVec_);
  
  stk::mesh::BucketVector const& locally_owned_node_bucket =
    bulkData_->get_buckets( stk::topology::NODE_RANK, s_locally_owned );
  
  for ( stk::mesh::BucketVector::const_iterator ib = locally_owned_node_bucket.begin();
        ib != locally_owned_node_bucket.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();

    // point to data
    const double * coords = stk::mesh::field_data(*coordinates_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get node
      stk::mesh::Entity node = b[k];

      // create the info
      OversetInfo *theInfo = new OversetInfo(node, nDim_);
      
      // push it back
      oversetInfoVec_.push_back(theInfo);
      
      // pointers to real data  
      const size_t offSet = k*nDim_;

      // fill in nodal coordinates
      for (int j = 0; j < nDim_; ++j ) {
        const double xj = coords[offSet+j];
        theInfo->nodalCoords_[j] = xj;
        localNodalCoords[j] = xj;
      }
    
      // define the ident
      stk::search::IdentProc<uint64_t,int> theIdent(bulkData_->identifier(node), NaluEnv::self().parallel_rank());

      // create the bounding point box and push back
      boundingPoint thePt(localNodalCoords, theIdent);
      boundingPointVec_.push_back(thePt);
    }
  }
}

//--------------------------------------------------------------------------
//-------- fringe_point_search ---------------------------------------------
//--------------------------------------------------------------------------
void
Overset::fringe_point_search()
{
  searchKeyPair_.clear();
  stk::search::coarse_search(boundingPointVec_, boundingElementBackgroundBoxVec_, searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair_);

  // we now have the coarse search for fringe nodes and backgroun bounding box

  // next steps are to ghost and to perform the fine search through isInElement
}

//--------------------------------------------------------------------------
//-------- set_data_on_inactive_part ---------------------------------------
//--------------------------------------------------------------------------
void
Overset::set_data_on_inactive_part()
{  
  // hack set element variables

  stk::mesh::Selector s_inactive = stk::mesh::Selector(*inActivePart_);

  // set nodal fields
  stk::mesh::BucketVector const& node_buckets = bulkData_->get_buckets( stk::topology::NODE_RANK, s_inactive );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * nodeIntersectedMesh = stk::mesh::field_data(*nodeIntersectedMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      nodeIntersectedMesh[k] = 1.0;
    }
  }
  
  // set element fields
  stk::mesh::BucketVector const& elem_buckets = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_inactive );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin() ;
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * elemIntersectedMesh = stk::mesh::field_data(*elemIntersectedMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      elemIntersectedMesh[k] = 1.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_data_on_overset_part ----------------------------------------
//--------------------------------------------------------------------------
void
Overset::set_data_on_overset_part()
{
  stk::mesh::Selector s_all_entities = stk::mesh::Selector(*volumePartVector_[1]);

  // set nodal fields
  stk::mesh::BucketVector const& node_buckets = bulkData_->get_buckets( stk::topology::NODE_RANK, s_all_entities );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * nodeOversetMesh = stk::mesh::field_data(*nodeOversetMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      nodeOversetMesh[k] = 6.0;
    }
  }
  
  // set element fields
  stk::mesh::BucketVector const& elem_buckets = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_all_entities );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin() ;
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * elemOversetMesh= stk::mesh::field_data(*elemOversetMesh_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      elemOversetMesh[k] = 6.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- output_results -----------------------------------------------
//--------------------------------------------------------------------------
void
Overset::output_results()
{
  ioBroker_->process_output_request(resultsFileIndex_, currentTime_);
}

} // namespace naluUnit
} // namespace Sierra

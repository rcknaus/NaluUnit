/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <superElement/SuperElement.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>

// edges and faces
#include <stk_mesh/base/CreateEdges.hpp>
#include <stk_mesh/base/CreateFaces.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// stk_io
#include <stk_io/StkMeshIoBroker.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// c++
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// SuperElement - unit test for super element
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SuperElement::SuperElement()
  : pOrder_(2),
    activateAura_(false),
    currentTime_(0.0),
    resultsFileIndex_(1),
    nDim_(2),
    metaData_(NULL),
    bulkData_(NULL),
    ioBroker_(NULL),
    nodeField_(NULL),
    coordinates_(NULL),
    originalPartName_("block_1"),
    originalSurfacePartName_("surface_1"),
    superElementPartName_("block_1_se"),
    superElementSurfacePartName_("surface_1_se"),
    promotedNodesPartName_("block_1_se_n"),
    edgePartName_("block_1_edges"),
    facePartName_("block_1_faces"),
    verboseOutput_(false),
    originalPart_(NULL),
    originalSurfacePart_(NULL),
    superElementPart_(NULL),
    superElementSurfacePart_(NULL),
    promotedNodesPart_(NULL),
    edgePart_(NULL),
    facePart_(NULL),
    numberOfEdges_(0),
    numberOfFaces_(0),
    numberOfElements_(0)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SuperElement::~SuperElement()
{
  delete bulkData_;
  delete metaData_;
  delete ioBroker_;
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void 
SuperElement::execute() 
{
  NaluEnv::self().naluOutputP0() << "Welcome to the SuperElement unit test" << std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "SuperElement Quad4 Unit Tests" << std::endl;
  NaluEnv::self().naluOutputP0() << "-----------------------------" << std::endl;

  stk::ParallelMachine pm = NaluEnv::self().parallel_comm();
  
  // news for mesh constructs
  metaData_ = new stk::mesh::MetaData();
  bulkData_ = new stk::mesh::BulkData(*metaData_, pm, activateAura_ ? stk::mesh::BulkData::AUTO_AURA : stk::mesh::BulkData::NO_AUTO_AURA);
  ioBroker_ = new stk::io::StkMeshIoBroker( pm );
  ioBroker_->set_bulk_data(*bulkData_);
  
  // deal with input mesh
  ioBroker_->add_mesh_database( "threeElemQuad4.g", stk::io::READ_MESH );

  ioBroker_->create_input_mesh();
  
  // safe to set nDim after create_input_mesh; not before
  nDim_ = metaData_->spatial_dimension();

  // check to make sure that we are supporting
  if ( nDim_ > 2 || pOrder_ > 2 )
    throw std::runtime_error("Only 2D P=2 is now supported");
  
  // create the part that holds the super element topo (volume and surface)
  declare_super_part();
  declare_super_part_surface();

  declare_edge_part();
  declare_face_part();
  
  // register the fields
  register_fields();
  register_fields_surface();
  
  // populate bulk data
  ioBroker_->populate_bulk_data();

  // create the edges and faces on low order part; tmp part(s) to later delete
  create_edges();
  create_faces();
  
  // extract coordinates
  coordinates_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  
  // create the parent id maps
  size_of_edges();
  size_of_faces();
  size_of_elements();

  // create nodes
  create_nodes();
  
  // for edges that have multiple processor owners, consolidate ids
  consolidate_edge_node_ids_at_boundaries();
  
  // create the element
  create_elements();
  const bool tryMe = false;
  if ( tryMe ) {
    create_elements_surface();
  }
  else {
    if ( verboseOutput_ ) {
      NaluEnv::self().naluOutputP0() << "..Unit Test Notes: " << std::endl;
      NaluEnv::self().naluOutputP0() << "....Need to figure out how to provide the super element topo with # sides, faces, etc for surface creation"
                                     << std::endl;
    }
  }
  
  // delete the edges and faces
  delete_edges();
  delete_faces();
  
  // deal with output mesh
  set_output_fields();
  
  // initialize nodal fields; define selector (locally owned and ghosted)
  initialize_fields();

  // output results
  output_results();
}

//--------------------------------------------------------------------------
//-------- declare_super_part ----------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::declare_super_part()
{
  // set nodes per element; assume quad or hex
  int nodesPerElem = (pOrder_ + 1)*(pOrder_ + 1);
  if ( nDim_ > 2)
    nodesPerElem *= (pOrder_ + 1);

  // create the super topo; how to assign number of sides, faces, etc?
  stk::topology superElemTopo = stk::create_superelement_topology(nodesPerElem);
  
  // two ways to create the part... WIP for doOld...
  const bool doOld = false;
  if ( doOld ) {
    // declare part with superTopo
    superElementPart_ = &metaData_->declare_part_with_topology(superElementPartName_, superElemTopo);
  }
  else {
    // declare part with element rank
    superElementPart_ = &metaData_->declare_part(superElementPartName_, stk::topology::ELEMENT_RANK);
    stk::mesh::set_topology(*superElementPart_, superElemTopo);
  }
 
  // we want this part to show up in the output mesh
  stk::io::put_io_part_attribute(*superElementPart_);
  
  // save off lower order part
  originalPart_ = metaData_->get_part(originalPartName_);
  
  // declare part for nodes
  promotedNodesPart_ = &metaData_->declare_part(promotedNodesPartName_, stk::topology::NODE_RANK);
}

//--------------------------------------------------------------------------
//-------- declare_super_part_surface --------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::declare_super_part_surface()
{
  // now deal with surface
  int nodesPerFace = pOrder_+1;
  if ( nDim_ > 2 )
    nodesPerFace *= (pOrder_+1);
    
  // create the super topo
  stk::topology superElemSurfaceTopo = stk::create_superelement_topology(nodesPerFace);
  
  // two ways to create the part... WIP for doOld...
  const bool doOld = false;
  if ( doOld ) {
    // declare part with superTopo
    superElementSurfacePart_ = &metaData_->declare_part_with_topology(superElementSurfacePartName_,   superElemSurfaceTopo);
  }
  else {
    // declare part with element rank
    superElementSurfacePart_ = &metaData_->declare_part(superElementSurfacePartName_,
                                                        metaData_->side_rank());
    stk::mesh::set_topology(*superElementSurfacePart_, superElemSurfaceTopo);
  }
    
  // we want this part to show up in the output mesh
  stk::io::put_io_part_attribute(*superElementSurfacePart_);
    
  // save off lower order part
  originalSurfacePart_ = metaData_->get_part(originalSurfacePartName_);
}

//--------------------------------------------------------------------------
//-------- declare_edge_part -----------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::declare_edge_part()
{
  edgePart_ = &metaData_->declare_part(edgePartName_, stk::topology::NODE_RANK);
}

//--------------------------------------------------------------------------
//-------- declare_face_part -----------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::declare_face_part()
{
  if ( nDim_ == 3 )
    facePart_ = &metaData_->declare_part(edgePartName_, stk::topology::FACE_RANK);
}

//--------------------------------------------------------------------------
//-------- create_edges ----------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_edges()
{
  stk::mesh::create_edges(*bulkData_, stk::mesh::Selector(*originalPart_), edgePart_);
}

//--------------------------------------------------------------------------
//-------- delete_edges ----------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::delete_edges()
{
  // complex delettion...

  // check to see how many edges remain in the original part now...
  size_t numberOfRemainingEdges = 0;
  // selector based on locally owned and shared edges
  stk::mesh::Selector s_edge = stk::mesh::Selector(*originalPart_);
  
  stk::mesh::BucketVector const& edge_remaining_buckets =
    bulkData_->get_buckets(stk::topology::EDGE_RANK, s_edge );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_remaining_buckets.begin();
        ib != edge_remaining_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();    
    // increment size
    numberOfRemainingEdges += length; 
  }
  
  // HACK... there should be two removed from the edge part; ten should live in the original part
  const bool testEdgeCount = numberOfRemainingEdges == 10 ? true : false;
  
  if (verboseOutput_ )
    NaluEnv::self().naluOutputP0() << "remaining edges: " << numberOfRemainingEdges << std::endl;
  
  if ( testEdgeCount )
    NaluEnv::self().naluOutputP0() << "Remaining Edge Count Test   PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Remaining Edge Count Test   FAILED" << std::endl;
}

//--------------------------------------------------------------------------
//-------- create_faces ------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_faces()
{
  if ( nDim_ == 3 )
    stk::mesh::create_faces(*bulkData_, stk::mesh::Selector(*originalPart_), facePart_);
}

//--------------------------------------------------------------------------
//-------- delete_faces ------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::delete_faces()
{
  // delete it; not yet
}

//--------------------------------------------------------------------------
//-------- size_of_edges ---------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::size_of_edges()
{ 
  // size edge count
  numberOfEdges_ = 0;

  // selector based on locally owned and shared edges
  stk::mesh::Selector s_edge = stk::mesh::Selector(*originalPart_);

  stk::mesh::BucketVector const& edge_buckets =
    bulkData_->get_buckets(stk::topology::EDGE_RANK, s_edge );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();    

    // increment size
    numberOfEdges_ += length;
      
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {           
      // finally, determine the the set of processors that this edge touches; defines commmunication pattern for nodes
      std::vector<int> sharedProcsEdge;
      bulkData_->comm_shared_procs({stk::topology::EDGE_RANK, bulkData_->identifier(b[k])}, sharedProcsEdge);
      std::sort(sharedProcsEdge.begin(), sharedProcsEdge.end());
      sharedProcsEdge_.push_back(sharedProcsEdge);
    }
  }

  if (verboseOutput_ )
    NaluEnv::self().naluOutputP0() << "size of edges: " << numberOfEdges_ << std::endl;

  const bool testEdge = numberOfEdges_ == 10 ? true : false;
  if ( testEdge )
    NaluEnv::self().naluOutputP0() << "Total Edge Count Test       PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Total Edge Count Test       FAILED" << std::endl;
}

//--------------------------------------------------------------------------
//-------- size_of_faces ---------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::size_of_faces()
{
  // size number of faces; not ready for 3D....
  numberOfFaces_ = 0;

  // define some common selectors
  stk::mesh::Selector s_face = stk::mesh::Selector(*originalPart_);

  stk::mesh::BucketVector const& face_buckets =
    bulkData_->get_buckets(stk::topology::FACE_RANK, s_face );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();    
    // increment size
    numberOfFaces_ += length;
  }
  
  if (verboseOutput_ )
    NaluEnv::self().naluOutputP0() << "size of faces: " << numberOfFaces_ << std::endl;

  if ( numberOfFaces_ > 0 )
    throw std::runtime_error("size_of_faces: greater than zero; 3D not ready for prime time");  
}

//--------------------------------------------------------------------------
//-------- size_of_elements ------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::size_of_elements()
{
  // size element count
  numberOfElements_ = 0;

  // define some common selectors; want locally owned here
  stk::mesh::Selector s_elem = metaData_->locally_owned_part()
    & stk::mesh::Selector(*originalPart_);

  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_elem );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    // increment size
    numberOfElements_ += length;   
  }

  if (verboseOutput_ )
    NaluEnv::self().naluOutputP0() << "size of elems: " << numberOfElements_ << std::endl;

  const bool testElem = numberOfElements_ == 3 ? true : false;
  if ( testElem )
    NaluEnv::self().naluOutputP0() << "Total Elem Count Test       PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Total Elem Count Test       FAILED" << std::endl;
}

//--------------------------------------------------------------------------
//-------- create_nodes ----------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_nodes()
{
  // count the number of promoted nodal ids required; based on parentIds size (which has been sorted)
  const int pM1Order = pOrder_ - 1;
  const int pElemFac = std::pow(pM1Order, nDim_);
  const int pEdgeFac = pM1Order;
  const int pFaceFac = pM1Order*pM1Order;
  
  const int numPromotedNodes
    = numberOfEdges_*pEdgeFac
    + numberOfFaces_*pFaceFac 
    + numberOfElements_*pElemFac;

  // okay, now ask
  bulkData_->modification_begin();

  // generate new ids; number of points is simple for now... all of the extra nodes from P=1 to P=2
  stk::mesh::EntityIdVector availableNodeIds(numPromotedNodes);
  bulkData_->generate_new_ids(stk::topology::NODE_RANK, numPromotedNodes, availableNodeIds);
  
  // declare the entity on this rank (rank is determined by calling declare_entity on this rank)
  for (int i = 0; i < numPromotedNodes; ++i) {
    stk::mesh::Entity theNode 
      = bulkData_->declare_entity(stk::topology::NODE_RANK, availableNodeIds[i], *promotedNodesPart_);
    promotedNodesVec_.push_back(theNode);
  }

  bulkData_->modification_end();
      
  // fill in std::map<stk::mesh::EntityIdVector, stk::mesh::Entity > parentNodesMap_
  int promotedNodesVecCount = 0;

  // edge selectors; locally owned and shared edges
  stk::mesh::Selector s_edge = stk::mesh::Selector(*originalPart_);

  stk::mesh::BucketVector const& edge_buckets =
    bulkData_->get_buckets(stk::topology::EDGE_RANK, s_edge );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {           
      
      // get edge and edge id
      stk::mesh::Entity edge = b[k];
      const stk::mesh::EntityId edgeId = bulkData_->identifier(edge);

      // extract node relations from the edge and node count
      stk::mesh::Entity const * edge_node_rels =  bulkData_->begin_nodes(edge);
      const int nodesPerEdge = b.num_nodes(k);

      // sanity check on number or nodes
      ThrowAssert( 2 == nodesPerEdge );

      // left, right and center node along the edge
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];
      stk::mesh::Entity nodeC = promotedNodesVec_[promotedNodesVecCount];
      
      const double * edgeCoordsL = stk::mesh::field_data(*coordinates_, nodeL);
      const double * edgeCoordsR = stk::mesh::field_data(*coordinates_, nodeR);
      double * edgeCoordsC = stk::mesh::field_data(*coordinates_, nodeC);
      
      // find mean distance between the nodes
      for ( int j = 0; j < nDim_; ++j )
        edgeCoordsC[j] = (edgeCoordsL[j] + edgeCoordsR[j])*0.5;
      
      // store off map
      parentEdgeNodesMap_[edgeId] = nodeC;
      
      ++promotedNodesVecCount;
    }
  }

  // fill in faces
  
  // element selectors; locally owned only
  stk::mesh::Selector s_elem = metaData_->locally_owned_part()
    & stk::mesh::Selector(*originalPart_);
  
  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_elem );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get element and element id
      stk::mesh::Entity elem = b[k];
      const stk::mesh::EntityId elemId = bulkData_->identifier(elem);
      
      // extract node relations
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);

      // iterate over the nodes in the element
      int numNodes = b.num_nodes(k);

      // extract element center noce
      stk::mesh::Entity nodeC = promotedNodesVec_[promotedNodesVecCount];
      double * elemCoordsC = stk::mesh::field_data(*coordinates_, nodeC);
    
      // hacked center coords calulation
      std::vector<double>tmpCoord(nDim_,0.0);
      for ( int ni = 0; ni < numNodes; ++ni ) {
        stk::mesh::Entity theNode = node_rels[ni];
        double * elemNodeCoords = stk::mesh::field_data(*coordinates_, theNode);
        for ( int i = 0; i < nDim_; ++i )
          tmpCoord[i] += elemNodeCoords[i]*0.25;
      }
      
      for ( int i = 0; i < nDim_; ++i )
        elemCoordsC[i] = tmpCoord[i];
      
      parentElemNodesMap_[elemId] = nodeC;
      
      ++promotedNodesVecCount;
    }  
  }
}

//--------------------------------------------------------------------------
//-------- consolidate_edge_node_ids_at_boundaries ------------------------
//--------------------------------------------------------------------------
void
SuperElement::consolidate_edge_node_ids_at_boundaries()
{
  // nothing
}
  
//--------------------------------------------------------------------------
//-------- create_elements -------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_elements()
{

  // define some common selectors; want locally owned here
  stk::mesh::Selector s_elem = metaData_->locally_owned_part()
    & stk::mesh::Selector(*originalPart_);
  
  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_elem );

  // elements and assign the new node connectivity
  bulkData_->modification_begin();
  
  // generate new ids; one per bucket loop
  stk::mesh::EntityIdVector availableElemIds(numberOfElements_);
  bulkData_->generate_new_ids(stk::topology::ELEM_RANK, numberOfElements_, availableElemIds);

  // generic iterator for parentNodesMap_ and placeholder for the found node
  std::map<stk::mesh::EntityId, stk::mesh::Entity>::iterator iterFindMap;
  stk::mesh::Entity foundNode;
  
  // declare id counter
  size_t availableElemIdCounter = 0;
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
         ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // define the vector that will hold the connected nodes for this element
      stk::mesh::EntityIdVector connectedNodesIdVec;
      
      // get element and element id
      stk::mesh::Entity elem = b[k];
      const stk::mesh::EntityId elemId = bulkData_->identifier(elem);
      
      // extract node relations amd mpde count
      stk::mesh::Entity const * elem_node_rels =  bulkData_->begin_nodes(elem);
      int numElemNodes = b.num_nodes(k);

      // first, standard nodes on the lower order elements
      for ( int ni = 0; ni < numElemNodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        connectedNodesIdVec.push_back(bulkData_->identifier(node));
      }
      
      // second, nodes in the center of the edges
      stk::mesh::Entity const * elem_edge_rels = bulkData_->begin_edges(elem);
      int numEdges = b.num_edges(k);
 
      for ( int ne = 0; ne < numEdges; ++ne ) {
         
        // extract the edge and edge id
        stk::mesh::Entity edge = elem_edge_rels[ne];
        stk::mesh::EntityId edgeId = bulkData_->identifier(edge);
        
        // find the edge centroid node(s)
        iterFindMap = parentEdgeNodesMap_.find(edgeId);
        if ( iterFindMap != parentEdgeNodesMap_.end() ) {
          foundNode = iterFindMap->second;
        }
        else {
          throw std::runtime_error("Could not find the node beloging to edge vector");
        }
        connectedNodesIdVec.push_back(bulkData_->identifier(foundNode));
      }
            
      // third, nodes in the center of the element faces
      
      // last, nodes in the center of the element; find the element centroid node(s)
      iterFindMap = parentElemNodesMap_.find(elemId);
      if ( iterFindMap != parentElemNodesMap_.end() ) {
        foundNode = iterFindMap->second;
      }
      else {
        throw std::runtime_error("Could not find the node beloging to element vector");
      }
      connectedNodesIdVec.push_back(bulkData_->identifier(foundNode));
      
      // all done with element, edge and face node connectivitoes; create the element
      stk::mesh::Entity theElem
        = stk::mesh::declare_element(*bulkData_, *superElementPart_,
                                     availableElemIds[availableElemIdCounter],
                                     connectedNodesIdVec);
      
      // push back to map
      superElementElemMap_[elemId] = theElem;
      
      availableElemIdCounter++;
    }
  }
  
  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- create_elements_surface ----------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_elements_surface()
{
  /* WIP
  // find total number of locally owned elements
  size_t numNewSurfaceElem = 0;
  
  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // generic iterator for parentElemMap_ and placeholder for the found element
  std::map<stk::mesh::EntityIdVector, stk::mesh::Entity>::iterator iterFindMap;
  stk::mesh::Entity thePromotedElement;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = metaData_->locally_owned_part()
  & stk::mesh::Selector(*originalSurfacePart_);
  
  stk::mesh::BucketVector const& face_buckets =
  bulkData_->get_buckets(metaData_->side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
       ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    numNewSurfaceElem += length;
  }
  
  // now loop over elements and assign the new node connectivity
  bulkData_->modification_begin();
  
  // generate new ids; one per bucket loop
  stk::mesh::EntityIdVector availableSurfaceElemIds(numNewSurfaceElem);
  bulkData_->generate_new_ids(metaData_->side_rank(), numNewSurfaceElem, availableSurfaceElemIds);
  
  // declare id counter
  size_t availableSurfaceElemIdCounter = 0;
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
       ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    // extract buckets face and element topology; not sure if we need this yet
    stk::topology thisBucketsTopo = b.topology();
    b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
    ThrowAssert ( parentTopo.size() == 1 );
    stk::topology theElemTopo = parentTopo[0];
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get face
      stk::mesh::Entity face = b[k];
      
      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = bulkData_->begin_elements(face);
      ThrowAssert( bulkData_->num_elements(face) == 1 );
      
      // get element; its face ordinal number and populate face_node_ordinal_vec
      stk::mesh::Entity element = face_elem_rels[0];
      const int faceOrdinal = bulkData_->begin_element_ordinals(face)[0];

      // vector to hold the element nodes
      stk::mesh::Entity const * elem_node_rels = bulkData_->begin_nodes(element);
      int numElemNodes = bulkData_->num_nodes(element);
      stk::mesh::EntityIdVector volumeCentroidVec(numElemNodes);
      for ( int ni = 0; ni < numElemNodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        volumeCentroidVec[ni] = bulkData_->identifier(node);
      }
      
      // sort the nodes and find the element from the superElement volume part
      std::sort(volumeCentroidVec.begin(), volumeCentroidVec.end());
      
      // find the edge centroid node(s)
      iterFindMap = superElementElemMap_.find(volumeCentroidVec);
      if ( iterFindMap != superElementElemMap_.end() ) {
        thePromotedElement = iterFindMap->second;
      }
      else {
        throw std::runtime_error("Could not find the element beloging to element vector");
      }
     
      // all done with element, edge and face node connectivitoes; create the element
      stk::mesh::declare_element_side(*bulkData_, availableSurfaceElemIds[availableSurfaceElemIdCounter],
                                      thePromotedElement, faceOrdinal, superElementSurfacePart_);
      availableSurfaceElemIdCounter++;
    }
  }
  
  bulkData_->modification_end();

*/
}
  
//--------------------------------------------------------------------------
//-------- register_fields -------------------------------------------------
//--------------------------------------------------------------------------
void 
SuperElement::register_fields()
{        
  // declare and put nodal field on part
  nodeField_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_field"));
    
  // put them on the part
  stk::mesh::put_field(*nodeField_, *superElementPart_);
  
  // declare and put element field on part
  elemField_ = &(metaData_->declare_field<GenericFieldType>(stk::topology::ELEM_RANK, "elem_field"));
  stk::mesh::put_field(*elemField_, *superElementPart_, 1);
}

//--------------------------------------------------------------------------
//-------- register_fields_surface -------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::register_fields_surface()
{
  // declare and put surface nodal field on part
  nodeSurfaceField_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_surface_field"));
  stk::mesh::put_field(*nodeSurfaceField_, *superElementSurfacePart_);
 
  // declare and put surface field on part
  stk::topology::rank_t sideRank = static_cast<stk::topology::rank_t>(metaData_->side_rank());
  surfaceField_ = &(metaData_->declare_field<GenericFieldType>(sideRank, "surface_field"));
  stk::mesh::put_field(*surfaceField_, *superElementSurfacePart_, 8);
}
  
//--------------------------------------------------------------------------
//-------- set_output_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::set_output_fields()
{  
  resultsFileIndex_ = ioBroker_->create_output_mesh( "superElement.e", stk::io::WRITE_RESULTS );
  ioBroker_->add_field(resultsFileIndex_, *nodeField_, nodeField_->name());
}
  
//--------------------------------------------------------------------------
//-------- initialize_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::initialize_fields()
{
  // just check on whether or not the nodes are all here on the superElementPart_; define gold standard for three element quad4 mesh
  const stk::mesh::EntityId goldElemNodalOrder[27] = {1, 2, 4, 8, 12, 13, 9, 16, 19,
                                                      8, 4, 5, 7, 9, 14, 10, 17, 20,
                                                      7, 5, 3, 6, 10, 15, 11, 18, 21};
  const stk::mesh::EntityId goldElemId[3] = {4,5,6};
  int goldElemIdCount = 0;
  int goldElemNodalOrderCount = 0;
  bool testElemIdPassed = true;
  bool testElemPassed = true;

  // define element selector
  stk::mesh::Selector s_elem = metaData_->locally_owned_part()
    & stk::mesh::Selector(*superElementPart_);

  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_elem );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get elem
      stk::mesh::Entity elem = b[k];
      const stk::mesh::EntityId elemId = bulkData_->identifier(elem);
  
      if ( elemId != goldElemId[goldElemIdCount] ) {
        testElemIdPassed = false;
        if ( verboseOutput_ )
          NaluEnv::self().naluOutputP0() << " elem id......FAILED " << elemId << " " << goldElemId[goldElemIdCount] << std::endl;
      }
      else {
        if ( verboseOutput_ )
          NaluEnv::self().naluOutputP0() << " elem id......PASSED " << elemId << " " << goldElemId[goldElemIdCount] << std::endl;
      }
      goldElemIdCount++;

      // number of nodes
      int num_nodes = b.num_nodes(k);

      // relations
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      
      if ( verboseOutput_ )
        NaluEnv::self().naluOutputP0() << "... number of nodes: " << num_nodes
                                       << " for element " << bulkData_->identifier(elem) << std::endl;

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // extract nodes
        double * nodalCoords = stk::mesh::field_data(*coordinates_, node);
      
        if ( verboseOutput_ )
          NaluEnv::self().naluOutputP0() << "Node id: " << bulkData_->identifier(node);
        if ( bulkData_->identifier(node) == goldElemNodalOrder[goldElemNodalOrderCount]) {
          if ( verboseOutput_ )
            NaluEnv::self().naluOutputP0() << " ......PASSED" << std::endl;
        }
        else {
          if ( verboseOutput_ )
            NaluEnv::self().naluOutputP0() << " ......FAILED" << std::endl;
          testElemPassed = false;
        }
        
        if ( verboseOutput_ ) {
          for ( int j = 0; j < nDim_; ++j )
            NaluEnv::self().naluOutputP0() << "     coords[" << j << "] " << nodalCoords[j] << std::endl;
        }
        
        // increment count
        goldElemNodalOrderCount++;
      }
    }
  }

  if ( testElemIdPassed )
    NaluEnv::self().naluOutputP0() << "Element Ids Test            PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Element Ids Test            FAILED" << std::endl;

  if ( testElemPassed )
    NaluEnv::self().naluOutputP0() << "Element Connectivities Test PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Element Connectivities Test FAILED" << std::endl;
    
  // now check nodes in the mesh based on super element part (same selector as above)
  size_t totalNumNodes = 0;
  size_t goldTotalNumNodes = 21;
  int goldNodalOrder[21] = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 4, 5, 7, 8, 3, 6, 2, 1};
 
  int goldNodalOrderCount = 0;
  bool testNodalPassed = true;
  
  // define node selector
  stk::mesh::Selector s_node = metaData_->locally_owned_part()
    & stk::mesh::Selector(*superElementPart_);

  stk::mesh::BucketVector const& node_buckets =
  bulkData_->get_buckets(stk::topology::NODE_RANK, s_node );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    totalNumNodes += length;
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get node
      stk::mesh::Entity node = b[k];
      
      if ( verboseOutput_ )
        NaluEnv::self().naluOutputP0() << "node identifier: " << bulkData_->identifier(node) << std::endl;
      
      if ( bulkData_->identifier(node) == goldNodalOrder[goldNodalOrderCount] ) {
        if ( verboseOutput_ )
          NaluEnv::self().naluOutputP0() << " ......PASSED" << std::endl;
      }
      else {
        if (verboseOutput_ )
          NaluEnv::self().naluOutputP0() << " ......FAILED" << std::endl;
        testNodalPassed = false;
      }
      
      // increment count
      goldNodalOrderCount++;
    }
  }
  
  if ( totalNumNodes != goldTotalNumNodes )
    testNodalPassed = false;
  
  if ( testNodalPassed )
    NaluEnv::self().naluOutputP0() << "Nodal iteration Test        PASSED" << std::endl;
  else 
    NaluEnv::self().naluOutputP0() << "Nodal iteration Test        FAILED" << std::endl;
  NaluEnv::self().naluOutputP0() << std::endl;
}

//--------------------------------------------------------------------------
//-------- output_results --------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::output_results()
{
  ioBroker_->process_output_request(resultsFileIndex_, currentTime_);
}

} // namespace naluUnit
} // namespace Sierra

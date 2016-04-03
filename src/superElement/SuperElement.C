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
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>

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
    superElementPartName_("block_1_se"),
    promotedNodesPartName_("block_1_se_n")
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
  
  // create the part that holds the super element topo; 
  declare_super_part();
  
  // register the fields
  register_fields();
  
  // populate bulk data
  ioBroker_->populate_bulk_data();
  
  // extract coordinates
  coordinates_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  
  // create nodes
  create_nodes();
  
  // create the element
  create_elements();
  
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

  // create the super topo
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
//-------- create_nodes ----------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_nodes()
{
  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = metaData_->locally_owned_part()
    & stk::mesh::Selector(*originalPart_);

  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    // extract buckets topology
    stk::topology thisBucketsTopo = b.topology();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get elem
      stk::mesh::Entity elem = b[k];

      // first, element center node
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);

      // vector to hold the element nodes
      int numNodes = b.num_nodes(k);
      stk::mesh::EntityIdVector volumeCentroidVec(numNodes);
      for ( int ni = 0; ni < numNodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];
        volumeCentroidVec[ni] = bulkData_->identifier(node);
      }

      // sort the nodes and push back
      std::sort(volumeCentroidVec.begin(), volumeCentroidVec.end());
      parentElemIds_.push_back(volumeCentroidVec);
      
      // second, nodes in the center of the edges
      const int numEdges = thisBucketsTopo.num_edges();
      for ( int ne = 0; ne < numEdges; ++ne ) {

        // although we are expecting a P=1 element, let's make this general for now
        const int nodesPerEdge = thisBucketsTopo.edge_topology().num_nodes();
        
        // extract the local element ids for this edge on this element
        std::vector<int> edgeNodeOrdinals(nodesPerEdge);
        thisBucketsTopo.edge_node_ordinals(ne,&edgeNodeOrdinals[0]);

        // vector to hold the edge nodes
        stk::mesh::EntityIdVector edgeCentroidVec(nodesPerEdge);
        for ( int ni = 0; ni < nodesPerEdge; ++ni ) {
          // extract local node id
          const int thisNode = edgeNodeOrdinals[ni];
          // extract node
          stk::mesh::Entity node = node_rels[thisNode];
          edgeCentroidVec[ni] = bulkData_->identifier(node);
        }

        // sort the nodes and push back
        std::sort(edgeCentroidVec.begin(), edgeCentroidVec.end());
        parentEdgeIds_.push_back(edgeCentroidVec);
      }
      
      // get number of faces
      const int numFaces = thisBucketsTopo.num_faces();
      for ( int nf = 0; nf < numFaces; ++nf ) {

        // although we are expecting quads or hexes (all nodes per face are the same) make this general for now
        const int nodesPerFace = thisBucketsTopo.face_topology(nf).num_nodes();
        std::vector<int> faceNodeOrdinals(nodesPerFace);
        thisBucketsTopo.edge_node_ordinals(nf,&faceNodeOrdinals[0]);
          
        // vector to hold face nodes
        stk::mesh::EntityIdVector faceCentroidVec;
        for ( int ni = 0; ni < nodesPerFace; ++ni ) {
          // extract local node id
          const int thisNode = faceNodeOrdinals[ni];
          // extract node
          stk::mesh::Entity node = node_rels[thisNode];
          faceCentroidVec[ni] = bulkData_->identifier(node);
        }

        // sort the nodes and push back
        std::sort(faceCentroidVec.begin(), faceCentroidVec.end());
        parentFaceIds_.push_back(faceCentroidVec);
      }
    }
  }

  // sort the full parentId vector. At this point; we may have duplicate entries
  std::sort(parentElemIds_.begin(), parentElemIds_.end());
  std::sort(parentEdgeIds_.begin(), parentEdgeIds_.end());
  std::sort(parentFaceIds_.begin(), parentFaceIds_.end());

  // prune for unique pairs
  std::vector<stk::mesh::EntityIdVector>::iterator pruneElemIdsIter
    = std::unique(parentElemIds_.begin(), parentElemIds_.end());
  std::vector<stk::mesh::EntityIdVector>::iterator pruneEdgeIdsIter
    = std::unique(parentEdgeIds_.begin(), parentEdgeIds_.end());
  std::vector<stk::mesh::EntityIdVector>::iterator pruneFaceIdsIter
    = std::unique(parentFaceIds_.begin(), parentFaceIds_.end());

  // now erase
  parentElemIds_.erase(pruneElemIdsIter, parentElemIds_.end());
  parentEdgeIds_.erase(pruneEdgeIdsIter, parentEdgeIds_.end());
  parentFaceIds_.erase(pruneFaceIdsIter, parentFaceIds_.end());

  // count the number of promoted nodal ids required; based on parentIds size (which has been sorted)
  const int pM1Order = pOrder_ - 1;
  const int pElemFac = std::pow(pM1Order, nDim_);
  const int pEdgeFac = pM1Order;
  const int pFaceFac = pM1Order*pM1Order;
  
  const int numPromotedNodes
    = parentElemIds_.size()*pElemFac
    + parentEdgeIds_.size()*pEdgeFac
    + parentFaceIds_.size()*pFaceFac;

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
  for ( size_t k = 0; k < parentEdgeIds_.size(); ++k ) {
    stk::mesh::EntityIdVector &theEntIdVec = parentEdgeIds_[k];
    // extract node and fill in coordinate
    stk::mesh::Entity edgeNode = promotedNodesVec_[promotedNodesVecCount];
    double * edgeCoords = stk::mesh::field_data(*coordinates_, edgeNode);
    
    // find mean distance between the nodes
    std::vector<double>tmpCoord(nDim_,0.0);
    for ( size_t j = 0; j < theEntIdVec.size(); ++j ) {
      stk::mesh::Entity theNode = bulkData_->get_entity(stk::topology::NODE_RANK, theEntIdVec[j]);
      double * edgeNodeCoords = stk::mesh::field_data(*coordinates_, theNode);
      for ( int i = 0; i < nDim_; ++i )
        tmpCoord[i] += edgeNodeCoords[i]*0.5;
    }
    for ( int i = 0; i < nDim_; ++i )
      edgeCoords[i] = tmpCoord[i];
    
    parentEdgeNodesMap_[theEntIdVec] = edgeNode;
    ++promotedNodesVecCount;
  }

  for ( size_t k = 0; k < parentFaceIds_.size(); ++k ) {
    stk::mesh::EntityIdVector &theEntIdVec = parentFaceIds_[k];
    
    // extract node and fill in coordinate
    stk::mesh::Entity faceNode = promotedNodesVec_[promotedNodesVecCount];
    double * faceCoords = stk::mesh::field_data(*coordinates_, faceNode);
    
    // find mean distance between the nodes
    std::vector<double>tmpCoord(nDim_,0.0);
    for ( size_t j = 0; j < theEntIdVec.size(); ++j ) {
      stk::mesh::Entity theNode = bulkData_->get_entity(stk::topology::NODE_RANK, theEntIdVec[j]);
      double * faceNodeCoords = stk::mesh::field_data(*coordinates_, theNode);
      for ( int i = 0; i < nDim_; ++i )
        tmpCoord[i] += faceNodeCoords[i]*0.25;
    }
    for ( int i = 0; i < nDim_; ++i )
      faceCoords[i] = tmpCoord[i];

    parentFaceNodesMap_[theEntIdVec] = faceNode;
    ++promotedNodesVecCount;
  }

  for ( size_t k = 0; k < parentElemIds_.size(); ++k ) {
    stk::mesh::EntityIdVector &theEntIdVec = parentElemIds_[k];
    // extract node and fill in coordinate
    stk::mesh::Entity elemNode = promotedNodesVec_[promotedNodesVecCount];
    double * elemCoords = stk::mesh::field_data(*coordinates_, elemNode);
    
    // find mean distance between the nodes
    std::vector<double>tmpCoord(nDim_,0.0);
    for ( size_t j = 0; j < theEntIdVec.size(); ++j ) {
      stk::mesh::Entity theNode = bulkData_->get_entity(stk::topology::NODE_RANK, theEntIdVec[j]);
      double * elemNodeCoords = stk::mesh::field_data(*coordinates_, theNode);
      for ( int i = 0; i < nDim_; ++i )
        tmpCoord[i] += elemNodeCoords[i]*0.25;
    }
    for ( int i = 0; i < nDim_; ++i )
      elemCoords[i] = tmpCoord[i];

    parentElemNodesMap_[theEntIdVec] = elemNode;
    ++promotedNodesVecCount;
  }
  
}

//--------------------------------------------------------------------------
//-------- create_elements -------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_elements()
{

  // find total number of locally owned elements
  size_t numNewElem = 0;
  
  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = metaData_->locally_owned_part()
    & stk::mesh::Selector(*originalPart_);
    
  stk::mesh::BucketVector const& elem_buckets =
  bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    numNewElem += length;
  }
  
  // now loop over elements and assign the new node connectivity
  bulkData_->modification_begin();
  
  // generate new ids; one per bucket loop
  stk::mesh::EntityIdVector availableElemIds(numNewElem);
  bulkData_->generate_new_ids(stk::topology::ELEM_RANK, numNewElem, availableElemIds);

  // generic iterator for parentNodesMap_ and placeholder for the found node
  std::map<stk::mesh::EntityIdVector, stk::mesh::Entity>::iterator iterFindMap;
  stk::mesh::Entity foundNode;

  // declare id counter
  size_t availableElemIdCounter = 0;
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
         ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    
    // extract buckets topology
    stk::topology thisBucketsTopo = b.topology();
        
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
            
      // get elem
      stk::mesh::Entity elem = b[k];
      
      // define the vector that will hold the connected nodes for this element
      stk::mesh::EntityIdVector connectedNodesIdVec;
      
      // extract node relations for this element
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      
      // first, standard nodes on the lower order element
      int numNodes = b.num_nodes(k);
      for ( int ni = 0; ni < numNodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];
        connectedNodesIdVec.push_back(bulkData_->identifier(node));
      }
      
      // second, nodes in the center of the edges
      const int numEdges = thisBucketsTopo.num_edges();
      for ( int ne = 0; ne < numEdges; ++ne ) {
        // although we are expecting a P=1 element, let's make this general for now
        const int nodesPerEdge = thisBucketsTopo.edge_topology().num_nodes();
                
        // extract the local element ids for this edge on this element
        std::vector<int> edgeNodeOrdinals(nodesPerEdge);
        thisBucketsTopo.edge_node_ordinals(ne,&edgeNodeOrdinals[0]);
                
        // vector to hold the edge nodes
        stk::mesh::EntityIdVector edgeCentroidVec(nodesPerEdge);
        for ( int ni = 0; ni < nodesPerEdge; ++ni ) {
          // extract local node id
          const int thisNodeId = edgeNodeOrdinals[ni];
          // extract node
          stk::mesh::Entity node = node_rels[thisNodeId];
          edgeCentroidVec[ni] = bulkData_->identifier(node);
        }
                
        // sort the nodes
        std::sort(edgeCentroidVec.begin(), edgeCentroidVec.end());
        // find the edge centroid node(s)
        iterFindMap = parentEdgeNodesMap_.find(edgeCentroidVec);
        if ( iterFindMap != parentEdgeNodesMap_.end() ) {
          foundNode = iterFindMap->second;
        }
        else {
          throw std::runtime_error("Could not find the node beloging to edge vector");
        }
        connectedNodesIdVec.push_back(bulkData_->identifier(foundNode));
      }
            
      // third, nodes in the center of the element faces
      const int numFaces = thisBucketsTopo.num_faces();
      for ( int nf = 0; nf < numFaces; ++nf ) {
        // although we are expecting quads or hexes (all nodes per face are the same) make this general for now
        const int nodesPerFace = thisBucketsTopo.face_topology(nf).num_nodes();
        std::vector<int> faceNodeOrdinals(nodesPerFace);
        thisBucketsTopo.edge_node_ordinals(nf,&faceNodeOrdinals[0]);
                
        // vector to hold face nodes
        stk::mesh::EntityIdVector faceCentroidVec;
        for ( int ni = 0; ni < nodesPerFace; ++ni ) {
          // extract local node id
          const int thisNode = faceNodeOrdinals[ni];
          // extract node
          stk::mesh::Entity node = node_rels[thisNode];
          faceCentroidVec[ni] = bulkData_->identifier(node);
        }
                
        // sort the nodes and push back
        std::sort(faceCentroidVec.begin(), faceCentroidVec.end());

        // find the FACE centroid node(s)
        iterFindMap = parentFaceNodesMap_.find(faceCentroidVec);
        if ( iterFindMap != parentFaceNodesMap_.end() ) {
          foundNode = iterFindMap->second;
        }
        else {
          throw std::runtime_error("Could not find the node beloging to face vector");
        }
        connectedNodesIdVec.push_back(bulkData_->identifier(foundNode));
      }
      
      // last, nodes in the center of the element
      stk::mesh::EntityIdVector volumeCentroidVec(numNodes);
      for ( int ni = 0; ni < numNodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];
        volumeCentroidVec[ni] = bulkData_->identifier(node);
      }
        
      // sort the nodes
      std::sort(volumeCentroidVec.begin(), volumeCentroidVec.end());
      
      // find the element centroid node(s)
      iterFindMap = parentElemNodesMap_.find(volumeCentroidVec);
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
      availableElemIdCounter++;
    }
  }

  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- register_fields -------------------------------------------------
//--------------------------------------------------------------------------
void 
SuperElement::register_fields()
{        
  // register nodal fields
  nodeField_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_field"));
    
  // put them on the part
  stk::mesh::put_field(*nodeField_, *originalPart_);
  stk::mesh::put_field(*nodeField_, *superElementPart_);
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
  int goldNodalOrder[27] = {1, 2, 4, 8, 9, 11, 15, 10, 19,
                          8, 4, 5, 7, 15, 14, 16, 18, 21,
                          7, 5, 3, 6, 16, 12, 13, 17, 20};
  int goldNodalOrderCount = 0;
  bool testPasses = true;
  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = metaData_->locally_owned_part()
    & stk::mesh::Selector(*superElementPart_);

  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get elem
      stk::mesh::Entity elem = b[k];

      // number of nodes
      int num_nodes = b.num_nodes(k);

      // relations
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      
      NaluEnv::self().naluOutputP0() << "... number of nodes: " << num_nodes 
                                     << " for element " << bulkData_->identifier(elem) << std::endl;

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // extract nodes
        double * nodalCoords = stk::mesh::field_data(*coordinates_, node);
      
        NaluEnv::self().naluOutputP0() << "Node id: " << bulkData_->identifier(node);
        if ( bulkData_->identifier(node) == goldNodalOrder[goldNodalOrderCount]) {
          NaluEnv::self().naluOutputP0() << " ......PASSED" << std::endl;
        }
        else {
          NaluEnv::self().naluOutputP0() << " ......FAILED" << std::endl;
          testPasses = false;
        }
        
        for ( int j = 0; j < nDim_; ++j )
          NaluEnv::self().naluOutputP0() << "     coords[" << j << "] " << nodalCoords[j] << std::endl;
        // increment count
        goldNodalOrderCount++;
      }
    }
  }
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "SuperElement Quad4" << std::endl;
  NaluEnv::self().naluOutputP0() << "------------------" << std::endl;
  if ( testPasses )
    NaluEnv::self().naluOutputP0() << "Test PASSED" << std::endl;
  else
    NaluEnv::self().naluOutputP0() << "Test FAILED" << std::endl;
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

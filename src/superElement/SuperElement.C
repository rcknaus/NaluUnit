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
  : activateAura_(false),
    currentTime_(0.0),
    resultsFileIndex_(1),
    nDim_(2),
    metaData_(NULL),
    bulkData_(NULL),
    ioBroker_(NULL),
    nodeField_(NULL),
    coordinates_(NULL)
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
  NaluEnv::self().naluOutputP0() << "Welcome to the SuperE unit test";

   stk::ParallelMachine pm = NaluEnv::self().parallel_comm();
  
   // news for mesh constructs
   metaData_ = new stk::mesh::MetaData();
   bulkData_ = new stk::mesh::BulkData(*metaData_, pm, activateAura_ ? stk::mesh::BulkData::AUTO_AURA : stk::mesh::BulkData::NO_AUTO_AURA);
   ioBroker_ = new stk::io::StkMeshIoBroker( pm );
   ioBroker_->set_bulk_data(*bulkData_);

   // deal with input mesh
   ioBroker_->add_mesh_database( "oneElemQuad4.g", stk::io::READ_MESH );
   ioBroker_->create_input_mesh();

   // create the part that holds the super element topo; 
   declare_super_part();

   // register the fields
   register_fields();

   // populate bulk data
   ioBroker_->populate_bulk_data();

   // create nodes
   create_nodes();

   // create the element
   create_elements();

   // deal with output mesh
   set_output_fields();

   // safe to set nDim
   nDim_ = metaData_->spatial_dimension();
 
   // extract coordinates
   coordinates_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

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
  // deal with parts
  std::string partName = "block_1";
  std::string partNameSE = partName + "_p" ;

  // hack for quad9
  const int nodesPerElem = 9;

  // create the super topo
  stk::topology superTopo = stk::create_superelement_topology(nodesPerElem);
  
  // declare part with superTopo
  superElementPart_ = &metaData_->declare_part_with_topology(partNameSE, superTopo);
  
  // save off lower order part
  lowerOrderPart_ = metaData_->get_part(partName);
  
  // declare part for nodes
  superElementAugmentedNodePart_ = &metaData_->declare_part(partNameSE + "_n", stk::topology::NODE_RANK);
}

//--------------------------------------------------------------------------
//-------- create_nodes ----------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_nodes()
{
  bulkData_->modification_begin();

  // generate new ids; number of points is simple for now... all of the extra nodes from P=1 to P=2
  const int numNodes = 5;
  stk::mesh::EntityIdVector availableNodeIds(numNodes);
  bulkData_->generate_new_ids(stk::topology::NODE_RANK, numNodes, availableNodeIds);

  // declare the entity on this rank (rank is determined by calling declare_entity on this rank)
  for (int i = 0; i < numNodes; ++i) {
    stk::mesh::Entity theNode 
      = bulkData_->declare_entity(stk::topology::NODE_RANK, availableNodeIds[i], *superElementAugmentedNodePart_);
    superElementAugmentedNodesVec_.push_back(theNode);
  }

  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- initialize_node_id_vec --------------------------------
//--------------------------------------------------------------------------
void
SuperElement::initialize_node_id_vec()
{

  // first, add nodal ids for standard lower order element

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = metaData_->locally_owned_part()
    & stk::mesh::Selector(*lowerOrderPart_);

  stk::mesh::BucketVector const& elem_buckets =
    bulkData_->get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get elem
      stk::mesh::Entity elem = b[k];

      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // set connected nodes
        connectedNodesIdVec_.push_back(bulkData_->identifier(node));
      }
    }
  }

  // now, add super element augmented nodes
  for ( size_t k = 0; k < superElementAugmentedNodesVec_.size(); ++k ) {
    connectedNodesIdVec_.push_back(bulkData_->identifier(superElementAugmentedNodesVec_[k]));
  }
}

//--------------------------------------------------------------------------
//-------- create_elements -------------------------------------------------
//--------------------------------------------------------------------------
void
SuperElement::create_elements()
{
  // save off vector of standard node ids
  initialize_node_id_vec();
  
  bulkData_->modification_begin();

  // generate new ids; number of points is simple for now... all of the extra nodes from P=1 to P=2
  const int numElem = 1;
  stk::mesh::EntityIdVector availableElemIds(numElem);
  bulkData_->generate_new_ids(stk::topology::ELEM_RANK, numElem, availableElemIds);

  // declare the entity on this rank (rank is determined by calling declare_entity on this rank)
  stk::mesh::PartVector partVec;
  partVec.push_back(superElementPart_);
  for (int i = 0; i < numElem; ++i) {
    stk::mesh::Entity theElem 
      = stk::mesh::declare_element(*bulkData_, *superElementPart_, availableElemIds[i], connectedNodesIdVec_);
  }

  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- register_fields -------------------------------------------------
//--------------------------------------------------------------------------
void 
SuperElement::register_fields()
{
  // extract the part(s)
  stk::mesh::Part *targetPart_1 = metaData_->get_part("block_1");
  stk::mesh::Part *targetPart_1_p = metaData_->get_part("block_1_p");
        
  // register nodal fields
  nodeField_ = &(metaData_->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "node_field"));
    
  // put them on the part
  stk::mesh::put_field(*nodeField_, *targetPart_1);
  stk::mesh::put_field(*nodeField_, *targetPart_1_p);
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
  // do something
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

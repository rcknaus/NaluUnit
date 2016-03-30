/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef SuperElement_h
#define SuperElement_h

// stk_mesh
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

// stk_search
#include <stk_search/BoundingBox.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/SearchMethod.hpp>

// STL
#include <vector>
#include <map>

// field types
typedef stk::mesh::Field<double>  ScalarFieldType;
typedef stk::mesh::Field<double, stk::mesh::Cartesian>  VectorFieldType;
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag>  GenericFieldType;

namespace stk {
  namespace io {
    class StkMeshIoBroker;
  }
  namespace mesh {
    class Part;
    class MetaData;
    class BulkData;
    typedef std::vector<Part*> PartVector;
    typedef std::vector<stk::mesh::EntityId> EntityIdVector;
    struct Entity;
  }
}

namespace sierra {
namespace naluUnit {

class SuperElement
{
public:

  // constructor/destructor
  SuperElement();
  ~SuperElement();

  void execute();

  // home for super part
  void declare_super_part();

  void create_nodes();

  void create_elements();

  // register nodal and elemental fields
  void register_fields();

  // declare output mesh and provide list of fields to send to this file
  void set_output_fields();

  // initialize nodal fields
  void initialize_fields();

  // provide output
  void output_results();

  // aura on/off
  const bool activateAura_;

  double currentTime_;
  size_t resultsFileIndex_;
  int nDim_;

  // meta, bulk and io
  stk::mesh::MetaData *metaData_;
  stk::mesh::BulkData *bulkData_;
  stk::io::StkMeshIoBroker *ioBroker_;

  // fields
  ScalarFieldType *nodeField_;
  VectorFieldType *coordinates_;

  // the original [volume] part of the lower order mesh, e.g., block_1, Topo::quad4
  std::string originalPartName_; 

  // the new volume part for the higher order mesh, e.g., block_1_SE, Topo::superElement
  std::string superElementPartName_; 

  // the set of nodes that are promoted
  std::string promotedNodesPartName_;

  // part associated with lower order standard element
  stk::mesh::Part *originalPart_;

  // part associated with super element
  stk::mesh::Part *superElementPart_;

  // in-transit part associated with augmented/promoted nodes
  stk::mesh::Part *promotedNodesPart_;

  // vector of new nodes
  std::vector<stk::mesh::Entity> promotedNodesVec_;

  // hold the unique vector of nodes for each element
  std::vector<stk::mesh::EntityIdVector> parentElemIds_;
  
  // hold the unique vector of nodes for each edge
  std::vector<stk::mesh::EntityIdVector> parentEdgeIds_;
  
  // hold the unique vector of nodes for each face
  std::vector<stk::mesh::EntityIdVector> parentFaceIds_;
  
  // create mapping of parent ids nodes to the new node
  std::map<stk::mesh::EntityIdVector, stk::mesh::Entity > parentElemNodesMap_;
  std::map<stk::mesh::EntityIdVector, stk::mesh::Entity > parentEdgeNodesMap_;
  std::map<stk::mesh::EntityIdVector, stk::mesh::Entity > parentFaceNodesMap_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

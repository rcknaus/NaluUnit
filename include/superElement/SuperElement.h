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

  void initialize_node_id_vec();
  
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

  std::string originalBlockName_;
  std::string superElementPartName_;
  std::string promotedNodesPartName_;

  // part associated with lower order standard element
  stk::mesh::Part *originalBlockPart_;

  // part associated with super element
  stk::mesh::Part *superElementPart_;

  // in-transit part associated with augmented/promoted nodes
  stk::mesh::Part *promotedNodesPart_;

  // vector of new nodes
  std::vector<stk::mesh::Entity> promotedNodesVec_;

  // vector of standard nodal ids
  stk::mesh::EntityIdVector connectedNodesIdVec_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Overset_h
#define Overset_h

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

// search types
typedef stk::search::IdentProc<uint64_t,int>  theKey;
typedef stk::search::Point<double> Point;
typedef stk::search::Box<double> Box;
typedef std::pair<Box,theKey> boundingElementBox;

namespace stk {
  namespace io {
    class StkMeshIoBroker;
  }
  namespace mesh {
    class Part;
    class MetaData;
    class BulkData;
    typedef std::vector<Part*> PartVector;
    struct Entity;
  }
}

namespace sierra {
namespace naluUnit {

class Overset
{
public:

  // constructor/destructor
  Overset();
  ~Overset();

  void execute();

  // register nodal and elemental fields
  void register_fields();

  // declare output mesh and provide list of fields to send to this file
  void set_output_fields();

  // initialize nodal and element fields
  void initialize_fields();

  // define the high level overset bounding box
  void define_overset_bounding_box();

  // find the intersection of the surface bounding box on underlying mesh
  void cut_surface();

  // define the underlying mesh set of bounding boxes
  void define_underlying_bounding_box();

  // process the coarse search; will provide the set of bounding boxes within the overset box
  void coarse_search();

  // create a part that will represent the inacative parts
  void create_inactive_part();

  // set data on inactive part
  void set_data_on_inactive_part();
  
  // set data on overset mesh
  void set_data_on_overset_part();

  // provide output
  void output_results();

  // data
  const bool activateAura_;
  const bool singleOversetBox_;
  const stk::search::SearchMethod searchMethod_;
  double currentTime_;
  size_t resultsFileIndex_;
  int nDim_;

  // meta, bulk and io
  stk::mesh::MetaData *metaData_;
  stk::mesh::BulkData *bulkData_;
  stk::io::StkMeshIoBroker *ioBroker_;

  // fields
  ScalarFieldType *nodeBackgroundMesh_;
  GenericFieldType *elemBackgroundMesh_;
  ScalarFieldType *nodeOversetMesh_;
  GenericFieldType *elemOversetMesh_;
  ScalarFieldType *nodeIntersectedMesh_;
  GenericFieldType *elemIntersectedMesh_;
  VectorFieldType *coordinates_;

  // part vector for the two blocks in the mesh
  stk::mesh::PartVector volumePartVector_;

  // search data structures
  std::vector<boundingElementBox> boundingElementOversetBoxVec_;
  std::vector<boundingElementBox> boundingElementUnderlyingBoxVec_;
  std::map<uint64_t, stk::mesh::Entity> searchElementMap_;

  /* save off product of search */
  std::vector<std::pair<theKey, theKey> > searchKeyPair_;
  
  // vector of elements intersected... will want to push to a part
  std::vector<stk::mesh::Entity > intersectedElementVec_;

  // part associated with inactive elements
  stk::mesh::Part *inActivePart_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

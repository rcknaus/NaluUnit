/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef SurfaceFields_h
#define SurfaceFields_h

// stk_mesh
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

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
    struct Entity;
  }
}

namespace sierra {
namespace naluUnit {

class SurfaceFields
{
public:

  // constructor/destructor
  SurfaceFields();
  ~SurfaceFields();

  void execute();
  
  // register nodal and elemental fields
  void register_fields();

  // declare output mesh and provide list of fields to send to this file
  void set_output_fields();

  // initialize nodal and element fields
  void initialize_fields();

  // make sure all fields are non-null
  void check_for_null();

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

  // subset the target surface part
  const bool doSubset_;

  // fields
  ScalarFieldType *normalHeatFlux_;

  // part vector for the three wall boundary surface in the mesh
  stk::mesh::PartVector surfacePartVector_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromoteElementRestartTest_h
#define PromoteElementRestartTest_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <stddef.h>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {
  class MasterElement;
  class PromoteElement;
  struct ElementDescription;
}
}

// field types
typedef stk::mesh::Field<double>  ScalarFieldType;
typedef stk::mesh::Field<int>  ScalarIntFieldType;
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag>  GenericFieldType;
typedef stk::mesh::Field<double, stk::mesh::Cartesian>  VectorFieldType;

namespace stk {
  namespace io {
    class StkMeshIoBroker;
  }
  namespace mesh {
    class BulkData;
    class MetaData;
    class Part;
    class Selector;

    typedef std::vector<Part*> PartVector;
  }
}

namespace sierra {
namespace naluUnit {
class MasterElement;
class PromotedElementIO;
class PromotedElement;
}  // namespace naluUnit
}  // namespace sierra

namespace sierra {
namespace naluUnit {


class PromoteElementRestartTest
{
public:
  // constructor/destructor
  PromoteElementRestartTest(std::string restartFileName, std::string outputFileName);
  ~PromoteElementRestartTest();

  void execute();

  void read_restart_mesh();
  void register_fields();
  void read_input_fields();
  void set_output_fields();
  void output_results();

  bool check_projected_nodal_gradient();
  void compute_projected_nodal_gradient();
  void compute_projected_nodal_gradient_interior(stk::mesh::Selector& selector);
  void compute_projected_nodal_gradient_boundary(stk::mesh::Selector& selector);

  std::unique_ptr<MasterElement>
  create_master_subcontrol_surface_element(const ElementDescription& elem);

  std::unique_ptr<MasterElement>
  create_master_boundary_element(const ElementDescription& elem);

  const std::string restartFileName_;
  const std::string outputFileName_;
  size_t restartFileIndex_;
  size_t resultsFileIndex_;
  double defaultFloatingPointTolerance_;
  unsigned nDim_;

  // meta, bulk, io, and promote element
  std::unique_ptr<stk::mesh::MetaData> metaData_;
  std::unique_ptr<stk::mesh::BulkData> bulkData_;
  std::unique_ptr<stk::io::StkMeshIoBroker> ioBroker_;

  // New element classes
  std::unique_ptr<ElementDescription> elem_;
  std::unique_ptr<MasterElement> meSCS_;
  std::unique_ptr<MasterElement> meBC_;
  std::unique_ptr<PromoteElement> promoteElement_;
  std::unique_ptr<PromotedElementIO> promoteIO_;

  //fields
  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;
  ScalarIntFieldType* sharedElems_;
  ScalarFieldType* q_;
  VectorFieldType* dqdx_;

  // part vectors
  stk::mesh::PartVector baseElemParts_;
  stk::mesh::PartVector superElemParts_;
  stk::mesh::PartVector promotedNodeParts_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

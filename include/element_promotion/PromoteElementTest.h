/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromoteElementTest_h
#define PromoteElementTest_h

#include <element_promotion/ElementDescription.h>
#include <element_promotion/PromotedElementIO.h>

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
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
  namespace mesh {
    class Part;
    class Selector;

    typedef std::vector<Part*> PartVector;
  }
}

namespace sierra {
namespace naluUnit {
class MasterElement;
class PromotedElementIO;
}  // namespace naluUnit
}  // namespace sierra

namespace sierra {
namespace naluUnit {


class PromoteElementTest
{
public:
  // constructor/destructor
  PromoteElementTest(
    int dimension,
    int order,
    std::string meshName,
    std::string quadType = "GaussLegendre"
  );
  ~PromoteElementTest();

  void execute();

  void setup_mesh();

  double timing_wall(double timeA, double timeB);

  void compute_dual_nodal_volume();
  void compute_projected_nodal_gradient();

  size_t count_nodes(stk::mesh::Selector selector);

  void register_fields();

  void set_output_fields();

  void initialize_fields();
  void initialize_scalar();

  void output_results();

  double determine_mesh_spacing();

  void dump_coords();

  std::unique_ptr<MasterElement>
  create_master_volume_element(const ElementDescription& elem);

  std::unique_ptr<MasterElement>
  create_master_subcontrol_surface_element(const ElementDescription& elem);

  std::unique_ptr<MasterElement>
  create_master_boundary_element(const ElementDescription& elem);

  void compute_dual_nodal_volume_interior(
    stk::mesh::Selector& selector);

  void compute_dual_nodal_volume_interior_SGL(
    stk::mesh::Selector& selector);

  void compute_projected_nodal_gradient_interior(
    stk::mesh::Selector& selector);

  void compute_projected_nodal_gradient_interior_SGL(
    stk::mesh::Selector& selector);

  void compute_projected_nodal_gradient_boundary(
    stk::mesh::Selector& selector);

  void compute_projected_nodal_gradient_boundary_SGL(
    stk::mesh::Selector& selector);

  bool check_node_count(unsigned polyOrder, unsigned originalNodeCount);

  std::string output_coords(stk::mesh::Entity node, unsigned dim);

  bool check_dual_nodal_volume();
  bool check_dual_nodal_volume_quad();
  bool check_dual_nodal_volume_hex();
  bool check_projected_nodal_gradient();

  const bool activateAura_;
  const double currentTime_;
  size_t resultsFileIndex_;
  size_t restartFileIndex_;
  const std::string meshName_;
  const double defaultFloatingPointTolerance_;

  // sets the scalar to 1. Otherwise, sets it equal to the
  // values for the heat conduction MMS
  bool linearScalarField_;
  unsigned nDim_;
  unsigned order_;
  bool outputTiming_;
  std::string quadType_;

  std::string elemType_;
  std::string coarseOutputName_;
  std::string fineOutputName_;
  std::string restartName_;


  // meta, bulk, io, and promote element
  std::unique_ptr<stk::mesh::MetaData> metaData_;
  std::unique_ptr<stk::mesh::BulkData> bulkData_;
  std::unique_ptr<stk::io::StkMeshIoBroker> ioBroker_;
  double timer_;

  // New element classes
  std::unique_ptr<PromoteElement> promoteElement_;
  std::unique_ptr<ElementDescription> elem_;
  std::unique_ptr<PromotedElementIO> promoteIO_;
  std::unique_ptr<MasterElement> meSCV_;
  std::unique_ptr<MasterElement> meSCS_;
  std::unique_ptr<MasterElement> meBC_;

  // fields
  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;
  ScalarFieldType* q_;
  ScalarIntFieldType* sharedElems_;
  VectorFieldType* dqdx_;
  stk::mesh::Field<double, stk::mesh::SimpleArrayTag>* tensorField_;

  // part vectors
  stk::mesh::PartVector originalPartVector_;
  stk::mesh::PartVector promotedPartVector_;
  stk::mesh::PartVector superElemPartVector_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

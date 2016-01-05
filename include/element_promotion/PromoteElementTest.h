/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromoteElementTest_h
#define PromoteElementTest_h

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
}  // namespace naluUnit
}  // namespace sierra

namespace sierra {
namespace naluUnit {


class PromoteElementTest
{
public:
  // constructor/destructor
  PromoteElementTest(int dimension, int order, std::string meshName);
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

  void compute_projected_nodal_gradient_interior(
    stk::mesh::Selector& selector);

  void compute_projected_nodal_gradient_boundary(
    stk::mesh::Selector& selector);

  bool check_node_count(unsigned polyOrder, unsigned originalNodeCount);
  bool is_near(double approx, double exact);
  bool is_near(const std::vector<double>& approx, const std::vector<double>& exact);
  bool is_near(
    const std::vector<double>& approx,
    const std::vector<double>& exact,
    double tolerance);
  bool check_interpolation_quad();
  bool check_interpolation_hex();
  bool check_derivative_quad();
  bool check_derivative_hex();
  bool check_volume_quadrature_quad();
  bool check_volume_quadrature_hex();
  bool check_lobatto();
  bool check_legendre();
  bool check_projected_nodal_gradient();
  double poly_val(std::vector<double> coeffs, double x);
  double poly_int(std::vector<double> coeffs,double xlower, double xupper);
  double poly_der(std::vector<double> coeffs, double x);

  std::string output_coords(stk::mesh::Entity node, unsigned dim);

  bool check_dual_nodal_volume();
  bool check_dual_nodal_volume_quad();
  bool check_dual_nodal_volume_hex();

  const bool activateAura_;
  const double currentTime_;
  size_t resultsFileIndex_;
  const std::string meshName_;
  const double floatingPointTolerance_;

  // sets the scalar to 1. Otherwise, sets it equal to the
  // values for the heat conduction MMS
  bool constScalarField_;
  unsigned nDim_;
  unsigned order_;

  std::string elemType_;
  std::string coarseOutputName_;
  std::string fineOutputName_;

  // meta, bulk, io, and promote element
  std::unique_ptr<stk::mesh::MetaData> metaData_;
  std::unique_ptr<stk::mesh::BulkData> bulkData_;
  std::unique_ptr<stk::io::StkMeshIoBroker> ioBroker_;

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
  ScalarIntFieldType* sharedElems_;
  ScalarFieldType* q_;
  VectorFieldType* dqdx_;

  // part vectors
  stk::mesh::PartVector originalPartVector_;
  stk::mesh::PartVector promotedPartVector_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

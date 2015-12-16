/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef PromoteElementTest_h
#define PromoteElementTest_h

#include <element_promotion/PromoteElement.h>
#include <nalu_make_unique.h>

// stk_mesh
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

// STL
#include <vector>
#include <map>
#include <memory>
#include <array>
#include <unordered_map>

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
    class Part;
    class MetaData;
    class BulkData;
    class Selector;
    typedef std::vector<Part*> PartVector;
    struct Entity;
  }
}

namespace sierra { namespace naluUnit { class MasterElement; } }
namespace sierra { namespace naluUnit { class PromotedElementIO; } }

namespace sierra {
namespace naluUnit {


class PromoteElementTest
{
public:
  // constructor/destructor
  PromoteElementTest(std::string elemType, std::string meshName);
  ~PromoteElementTest();

  void execute();

  void setup_mesh();

  double timing_wall(double timeA, double timeB);

  void compute_dual_nodal_volume();

  size_t count_nodes(stk::mesh::Selector selector);

  void register_fields();

  void set_output_fields();

  void initialize_fields();

  void output_results();

  double determine_mesh_spacing();

  void dump_coords();

  void compute_dual_nodal_volume_interior(MasterElement&& masterElement, stk::mesh::Selector selector);

  bool check_node_count(unsigned polyOrder, unsigned originalNodeCount);
  bool is_near(double approx, double exact);
  bool is_near(const std::vector<double>& approx, const std::vector<double>& exact);
  bool check_interpolation();
  bool check_derivative();
  bool check_volume_quadrature();
  double poly_val(std::vector<double> coefs, double x);
  double poly_int(std::vector<double> coeffs,double xlower, double xupper);
  double poly_der(std::vector<double> coeffs, double x);

  bool check_dual_nodal_volume();
  bool check_dual_nodal_volume_quad();
  bool check_dual_nodal_volume_hex();

  const bool activateAura_;
  double currentTime_;
  size_t resultsFileIndex_;
  const std::string elemType_;
  const std::string meshName_;
  const std::string coarseOutputName_;
  const std::string fineOutputName_;
  double floatingPointTolerance_;

  // meta, bulk, io, and promote element
  std::unique_ptr<stk::mesh::MetaData> metaData_;
  std::unique_ptr<stk::mesh::BulkData> bulkData_;
  std::unique_ptr<stk::io::StkMeshIoBroker> ioBroker_;
  unsigned nDim_;

  // New element classes
  std::unique_ptr<PromoteElement> promoteElement_;
  std::unique_ptr<ElementDescription> elem_;
  std::unique_ptr<PromotedElementIO> promoteIO_;

  // fields
  VectorFieldType* coordinates_;
  ScalarFieldType* dualNodalVolume_;
  ScalarIntFieldType* sharedElems_;

  // part vectors
  stk::mesh::PartVector originalPartVector_;
  stk::mesh::PartVector promotedPartVector_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

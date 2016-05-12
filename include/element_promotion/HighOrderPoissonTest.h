/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderPoissonTest_h
#define HighOrderPoissonTest_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseSolver.hpp>

#include <stddef.h>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {
  class MasterElement;
  class PromoteElement;
  class PromotedElementIO;
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

class HighOrderPoissonTest
{
public:
  // constructor/destructor
  HighOrderPoissonTest(std::string meshName = "test_meshes/hex8_2.g");
  ~HighOrderPoissonTest();

  void execute();

  void setup_mesh();

  void register_fields();

  void set_output_fields();

  void output_results();

  void initialize_fields();

  void output_banner();

  void setup_super_parts();

  void solve_poisson();

  std::unique_ptr<MasterElement>
  make_master_volume_element(const ElementDescription& elem);

  std::unique_ptr<MasterElement>
  make_master_subcontrol_surface_element(const ElementDescription& elem);

  std::unique_ptr<MasterElement>
  make_master_boundary_element(const ElementDescription& elem);

  bool check_solution();

  const std::string meshName_;
  const int order_;
  const bool outputTiming_;
  double timeCondense_;
  double timeInteriorUpdate_;

  std::string fineOutputName_;

  std::unique_ptr<ElementDescription> elem_;
  std::unique_ptr<PromotedElementIO> promoteIO_;

  // meta, bulk, io, and promote element
  std::unique_ptr<stk::mesh::MetaData> metaData_;
  std::unique_ptr<stk::mesh::BulkData> bulkData_;
  std::unique_ptr<stk::io::StkMeshIoBroker> ioBroker_;

  // fields
  VectorFieldType* coordinates_;
  ScalarFieldType* q_;
  ScalarIntFieldType* mask_;
  ScalarFieldType* qExact_;

  // part vectors
  stk::mesh::PartVector originalPartVector_;
  stk::mesh::PartVector superPartVector_;
  stk::mesh::PartVector superSidePartVector_;

  Teuchos::SerialDenseMatrix<int,double> lhs_;
  Teuchos::SerialDenseVector<int,double> rhs_;
  Teuchos::SerialDenseVector<int,double> delta_;
  std::map<stk::mesh::Entity, size_t> rowMap_;
  std::unique_ptr<MasterElement> meSCS_;
  std::unique_ptr<MasterElement> meSCV_;


private:
  void initialize_matrix();
  void assemble_poisson();
  void apply_dirichlet();
  void solve_matrix_equation();
  void update_field();
};

} // namespace naluUnit
} // namespace Sierra

#endif

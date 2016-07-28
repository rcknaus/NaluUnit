/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductPoissonTest_h
#define TensorProductPoissonTest_h

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

class TensorProductPoissonTest
{
public:
  // constructor/destructor
 TensorProductPoissonTest(
   std::string meshName = "test_meshes/tquad4_4.g",
   int order = 10,
   bool printTiming = true);
 ~TensorProductPoissonTest();

  void execute();
private:
  void setup_mesh();
  void register_fields();
  void set_output_fields();
  void output_results();
  void initialize_fields();
  void output_banner();
  void setup_super_parts();
  void perturb_coordinates(double elem_size, double fac);
  void solve_poisson();
  bool check_solution();
  void initialize_matrix();
  void assemble_poisson(unsigned pOrder);
  template<unsigned poly_order> void assemble_poisson();
  void sum_into_global(const size_t* indices, double* lhs_local, double* rhs_local, int length);
  void apply_dirichlet();
  void solve_matrix_equation();
  void update_field();

  const std::string meshName_;
  const int order_;
  const bool outputTiming_;
  double timeMainLoop_;
  double timeMetric_;
  double timeLHS_;
  double timeResidual_;
  double timeGather_;
  double timeVolumeMetric_;
  double timeVolumeSource_;
  size_t countAssemblies_;
  const double testTolerance_;
  const bool randomlyPerturbCoordinates_;

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
  ScalarFieldType* qExact_;
  ScalarFieldType* source_;

  // part vectors
  stk::mesh::PartVector originalPartVector_;
  stk::mesh::PartVector superPartVector_;
  stk::mesh::PartVector superSidePartVector_;

  Teuchos::SerialDenseMatrix<int,double> lhs_;
  Teuchos::SerialDenseVector<int,double> rhs_;
  Teuchos::SerialDenseVector<int,double> delta_;
  std::map<stk::mesh::Entity, size_t> rowMap_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef MasterElementHOTest_h
#define MasterElementHOTest_h

#include <element_promotion/ElementDescription.h>

#include <stddef.h>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {

class MasterElementHOTest
{
public:
  // constructor/destructor
  MasterElementHOTest(int dim, int maxOrder);
  ~MasterElementHOTest() = default;

  void execute();

  bool check_interpolation_quad();
  bool check_interpolation_hex();
  bool check_derivative_quad();
  bool check_derivative_hex();
  bool check_volume_quadrature_quad();
  bool check_volume_quadrature_quad_SGL();
  bool check_volume_quadrature_hex();
  bool check_volume_quadrature_hex_SGL();
  double poly_val(std::vector<double> coeffs, double x);
  double poly_int(std::vector<double> coeffs, double xlower, double xupper);
  double poly_der(std::vector<double> coeffs, double x);

  unsigned nDim_;
  unsigned polyOrder_;
  bool outputTiming_;
  std::unique_ptr<ElementDescription> elem_;
};

} // namespace naluUnit
} // namespace Sierra

#endif

/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/QuadratureRuleTest.h>

#include <NaluEnv.h>
#include <element_promotion/QuadratureRule.h>
#include <TestHelper.h>

#include <cmath>
#include <iostream>
#include <vector>


namespace sierra{
namespace naluUnit{

//--------------------------------------------------------------------------
// Checks the basic 1D quadrature rule
//--------------------------------------------------------------------------
void
QuadratureRuleTest::execute()
{
  NaluEnv::self().naluOutputP0() << "Quadrature Rule Unit Tests" << std::endl;
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;

  output_result("Legendre", check_legendre());
  output_result("Lobatto ",  check_lobatto());
  NaluEnv::self().naluOutputP0() << "-------------------------" << std::endl;
}
//--------------------------------------------------------------------------
bool
QuadratureRuleTest::check_lobatto()
{
  double tol = 5.0e-15; // needs to be pretty accurate

  bool testPassed = false;
  std::vector<double> abscissae;
  std::vector<double> weights;
  std::vector<double> exactX;
  std::vector<double> exactW;

  std::tie(abscissae,weights) = gauss_lobatto_legendre_rule(3);
  exactX = {-1.0, 0.0, +1.0};
  exactW = { 1.0/3.0, 4.0/3.0, 1.0/3.0 };

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_lobatto_legendre_rule(4);
  double xl0 = std::sqrt(5.0)/5.0;
  double xw0 = 5.0/6.0;
  double xw1 = 1.0/6.0;
  exactX = {-1.0, -xl0, +xl0, +1.0};
  exactW = { xw1, xw0, xw0, xw1 }; // sums to 2

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_lobatto_legendre_rule(5);
  xl0 = std::sqrt(21.0)/7.0;
  xw0 = 32.0/45.0;
  xw1 = 49.0/90.0;
  double xw2 = 1.0/10.0;
  exactX = {-1.0, -xl0, 0.0, xl0, +1.0};
  exactW = { xw2, xw1, xw0, xw1, xw2 }; // sums to 2

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_lobatto_legendre_rule(6);
  xl0 = std::sqrt((7.0-2.0*std::sqrt(7.0))/21.0);
  double xl1 = std::sqrt((7.0+2.0*std::sqrt(7.0))/21.0);
  xw0 = (14.0+std::sqrt(7.0))/30.0;
  xw1 = (14.0-std::sqrt(7.0))/30.0;
  xw2 = 1.0/15.0;
  exactX = {-1.0, -xl1, -xl0, xl0, +xl1, +1.0};
  exactW = { xw2, xw1, xw0, xw0, xw1, xw2 }; // sums to 2

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  testPassed = true;
  return testPassed;
}
//--------------------------------------------------------------------------
bool
QuadratureRuleTest::check_legendre()
{
  double tol = 1.0e-15; // needs to be pretty accurate

  bool testPassed = false;
  std::vector<double> abscissae;
  std::vector<double> weights;
  std::vector<double> exactX;
  std::vector<double> exactW;

  std::tie(abscissae,weights) = gauss_legendre_rule(2);
  exactX = {-std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };
  exactW = { 1.0, 1.0 };

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_legendre_rule(3);
  exactX = { -std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0) };
  exactW = { 5.0/9.0, 8.0/9.0,  5.0/9.0 };

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_legendre_rule(4);
  exactX = {
      -std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0)),
      -std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
      +std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
      +std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0))
  };

  exactW = {
      (18.0-std::sqrt(30.0))/36.0,
      (18.0+std::sqrt(30.0))/36.0,
      (18.0+std::sqrt(30.0))/36.0,
      (18.0-std::sqrt(30.0))/36.0
  };

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  std::tie(abscissae,weights) = gauss_legendre_rule(5);
  exactX = {
      -std::sqrt(245.0+14.0*std::sqrt(70.0))/21.0,
      -std::sqrt(245.0-14.0*std::sqrt(70.0))/21.0,
      0.0,
      +std::sqrt(245.0-14.0*std::sqrt(70.0))/21.0,
      +std::sqrt(245.0+14.0*std::sqrt(70.0))/21.0
  };

  exactW = {
      (322.0-13.0*std::sqrt(70.0))/900.0,
      (322.0+13.0*std::sqrt(70.0))/900.0,
      128.0/225.0,
      (322.0+13.0*std::sqrt(70.0))/900.0,
      (322.0-13.0*std::sqrt(70.0))/900.0
  };

  if (!is_near(abscissae,exactX,tol) || !is_near(weights,exactW,tol)) {
    return false;
  }

  testPassed = true;
  return testPassed;
}


} // namespace naluUnit
}  // namespace sierra

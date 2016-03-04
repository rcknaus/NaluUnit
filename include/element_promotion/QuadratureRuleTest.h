/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef QuadratureRuleTest_h
#define QuadratureRuleTest_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <stddef.h>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {

class QuadratureRuleTest
{
public:
  void execute();
  bool check_lobatto();
  bool check_legendre();
};

} // namespace naluUnit
} // namespace Sierra

#endif

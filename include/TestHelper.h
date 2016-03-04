/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TestHelper_h
#define TestHelper_h

#include <NaluEnv.h>

#include <stddef.h>
#include <ostream>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {

  template<typename Scalar = double> bool
  is_near(Scalar approx, Scalar exact,Scalar tol)
  {
    return (std::abs(approx-exact) < tol);
  }

  template<typename Container, typename Scalar = double> bool
  is_near(
    const Container& approx,
    const Container& exact,
    Scalar tol)
  {
    if (approx.size() != exact.size()) {
      return false;
    }
    for (unsigned j = 0; j < approx.size(); ++j) {
      if (!is_near(approx[j], exact[j], tol)) {
        return false;
      }
    }
    return true;
  }

  inline void
  output_result(std::string test_name, bool status)
  {
    if (status) {
      NaluEnv::self().naluOutputP0() << test_name << " TEST: PASSED " << std::endl;
    }
    else {
      NaluEnv::self().naluOutputP0() << test_name << " TEST: FAILED " << std::endl;
    }
  }


} // namespace naluUnit
} // namespace Sierra

#endif

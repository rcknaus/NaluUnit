/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TestHelper_h
#define TestHelper_h

#include <NaluEnv.h>

#include <stk_util/environment/ReportHandler.hpp>

#include <stddef.h>
#include <ostream>
#include <string>
#include <vector>
#include <limits>

namespace sierra {
namespace naluUnit {

  template<typename Scalar = double> bool
  is_near(Scalar approx, Scalar exact,Scalar tol)
  {
    return (std::abs(approx-exact) < tol);
  }

  template<typename Container> double
  max_error(const Container& approx, const Container& exact)
  {
    if (approx.size() != exact.size() || approx.empty()) {
      return std::numeric_limits<double>::max();
    }

    double err = -1.0;
    for (unsigned j = 0; j < approx.size(); ++j) {
      err = std::max(err, std::abs(approx[j]-exact[j]));
    }
    return err;
  }

  template<typename Container, typename Scalar = double> bool
  is_near(
    const Container& approx,
    const Container& exact,
    Scalar tol)
  {
    if (max_error(approx,exact) < tol) {
      return true;
    }
    return false;
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

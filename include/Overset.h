/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Overset_h
#define Overset_h

#include <vector>

namespace stk {
namespace mesh {
class Part;
typedef std::vector<Part*> PartVector;
}
}
namespace sierra{
namespace naluUnit{

class Overset
{
public:

  // constructor/destructor
  Overset();
  ~Overset();
};

} // namespace naluUnit
} // namespace Sierra

#endif

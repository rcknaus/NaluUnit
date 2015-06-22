/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <overset/OversetInfo.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace naluUnit{

//==========================================================================
// Class Definition
//==========================================================================
// OversetInfo - contains fringe point -> owning elements
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
OversetInfo::OversetInfo(
  stk::mesh::Entity node,
  const int nDim)
  : faceNode_(node),
    owningElement_(),
    bestX_(1.0e16),
    elemIsGhosted_(0)
{
  // resize stuff
  isoParCoords_.resize(nDim);
  nodalCoords_.resize(nDim);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
OversetInfo::~OversetInfo()
{
  // nothing to delete
}

} // namespace NaluUnit
} // namespace sierra

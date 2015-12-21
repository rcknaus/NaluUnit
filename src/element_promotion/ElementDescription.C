#include <element_promotion/ElementDescription.h>

#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <nalu_make_unique.h>

#include <stk_util/environment/ReportHandler.hpp>

#include <ext/alloc_traits.h>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace sierra {
namespace naluUnit {

std::unique_ptr<ElementDescription>
ElementDescription::create(std::string type)
{
  if (type == "Quad9") {
    std::vector<double> in_nodeLocs = { -1.0, 0.0, +1.0 };
    std::vector<double> in_scsLoc = { -std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };
    return make_unique<QuadMElementDescription>(in_nodeLocs,in_scsLoc);
  }
  if (type == "Quad16") {
    // symmetric mass matrix
    // I can't find a symmetric mass matrix for P > 3
    double xgll    = 0.4487053820572093009546164613323186035;
    double scsDist = 0.8347278713337825805263131558586123084;
    std::vector<double> in_nodeLocs = { -1.0, -xgll, +xgll, +1.0 };
    std::vector<double> in_scsLoc = { -scsDist, 0.0, scsDist };

    return make_unique<QuadMElementDescription>(in_nodeLocs,in_scsLoc);
  }
  if (type == "Quad25") {
     double xgll    = std::sqrt(21.0)/7.0;
     double scsDist1= std::sqrt(525.0-70.0*std::sqrt(30.0))/35.0;
     double scsDist2= std::sqrt(525.0+70.0*std::sqrt(30.0))/35.0;
     std::vector<double> in_nodeLocs = { -1.0, -xgll, 0.0, +xgll, +1.0 };
     std::vector<double> in_scsLoc = { -scsDist2, -scsDist1, scsDist1, scsDist2 };

     return make_unique<QuadMElementDescription>(in_nodeLocs,in_scsLoc);
  }
  if (type == "Quad36") {
    // Gauss-Lobatto nodes w/ Gauss-Legendre scs
     double xgll1    = std::sqrt((7.0-2.0*std::sqrt(7.0))/21.0);
     double xgll2    = std::sqrt((7.0+2.0*std::sqrt(7.0))/21.0);
     double scsDist1 = std::sqrt(245.0-14.0*std::sqrt(70.0))/21.0;
     double scsDist2 = std::sqrt(245.0+14.0*std::sqrt(70.0))/21.0;

     std::vector<double> in_nodeLocs = { -1.0, -xgll2, -xgll1, +xgll1, +xgll2, +1.0 };
     std::vector<double> in_scsLoc = { -scsDist2, -scsDist1, 0.0, +scsDist1, +scsDist2 };

     return make_unique<QuadMElementDescription>(in_nodeLocs,in_scsLoc);
  }
  if (type == "Hex27") {
    return make_unique<Hex27ElementDescription>();
  }

  throw std::runtime_error("Element type not implemented");
  return nullptr;
}

QuadMElementDescription::QuadMElementDescription(
  const std::vector<double> in_nodeLocs, const std::vector<double>& in_scsLoc)
  : ElementDescription()
{
  nodeLocs = in_nodeLocs;
  scsLoc = in_scsLoc;
  ThrowRequire(nodeLocs.size()-1 == scsLoc.size());

  polyOrder = nodeLocs.size()-1;
  nodes1D = nodeLocs.size();
  nodesPerElement = nodes1D*nodes1D;
  dimension = 2;
  numQuad = (polyOrder % 2 == 0) ? polyOrder/2 + 1 : (polyOrder+1)/2;
  useGLLGLL = false;

  set_node_maps(nodes1D);
  set_node_connectivity(nodes1D);
  set_node_locations(edgeNodeConnectivities, faceNodeConnectivities, nodeLocs);
  set_subelement_connectivity(nodes1D);

  quadrature = make_unique<TensorProductQuadratureRule>("GaussLegendre", polyOrder, scsLoc);
  basis = make_unique<LagrangeBasis>(inverseNodeMap, nodeLocs);
  basisBoundary = make_unique<LagrangeBasis>(inverseNodeMap1D,nodeLocs);
};
//--------------------------------------------------------------------------
void
QuadMElementDescription::set_node_maps(unsigned in_nodes1D)
{
  unsigned polyOrder = in_nodes1D-1;
  unsigned baseNodesPerElement = 4;

   nodeMap.assign(in_nodes1D*in_nodes1D,0);
   auto imap = [=] (unsigned i, unsigned j) { return (i+in_nodes1D*j);};

   // save a map in tensor product form for the node locations
   // proceed edge-by-edge, then face

   //forward
   unsigned nodeNumber = baseNodesPerElement;
   nodeMap[imap(0,0)] = 0;
   for (unsigned j = 1; j < polyOrder; ++j) {
     nodeMap[imap(j,0)] = nodeNumber;
     ++nodeNumber;
   }

   //forward
   nodeMap[imap(in_nodes1D-1,0)]  = 1;
   for (unsigned j = 1; j < polyOrder; ++j) {
     nodeMap[imap(in_nodes1D-1,j)] = nodeNumber;
     ++nodeNumber;
   }

  //reverse
   nodeMap[imap(in_nodes1D-1,in_nodes1D-1)] = 2;
   for (int j = polyOrder-1; j >0; --j) {
     nodeMap[imap(j, in_nodes1D-1)] = nodeNumber;
     ++nodeNumber;
   }

   //reverse
   nodeMap[imap(0,in_nodes1D-1)]  = 3;
   for (int j = polyOrder-1; j >0; --j) {
     nodeMap[imap(0,j)] =  nodeNumber;
     ++nodeNumber;
   }

   //base face node
   for (unsigned j = 1; j < polyOrder; ++j) {
     for (unsigned i = 1; i < polyOrder; ++i) {
       nodeMap[imap(i,j)] = nodeNumber;
       ++nodeNumber;
     }
   }

   //1D map
   nodeMap1D.resize(in_nodes1D);
   nodeMap1D[0] = 0;
   nodeMap1D[in_nodes1D-1] = 1;

   nodeNumber = 2;
   for (unsigned j = 1; j < polyOrder; ++j) {
     nodeMap1D[j] = nodeNumber;
     ++nodeNumber;
   }

   //inverse maps
   inverseNodeMap.resize(in_nodes1D*in_nodes1D);
   for (unsigned i = 0; i < in_nodes1D; ++i) {
     for (unsigned j = 0; j < in_nodes1D; ++j) {
       inverseNodeMap[tensor_product_node_map(i,j)] = {i, j};
     }
   }

   inverseNodeMap1D.resize(in_nodes1D);
   for (unsigned j = 0; j < in_nodes1D; ++j) {
     inverseNodeMap1D[tensor_product_node_map(j)] = { j };
   }
}
//--------------------------------------------------------------------------
void
QuadMElementDescription::set_node_connectivity(unsigned in_nodes1D)
{
  unsigned baseNodesPerElement = 4;
  unsigned nodeNumber = baseNodesPerElement;
  std::vector<std::vector<size_t>> baseEdgeNodes = { {0,1},{1,2},{2,3},{3,0} };
  std::vector<size_t> baseFaceNodes = {0, 1, 2, 3};

  faceNodeMap.resize(4);
  unsigned faceOrdinal = 0;
  for (auto& baseEdge : baseEdgeNodes) {
    std::vector<size_t> nodesToAdd(in_nodes1D-2);
    for (unsigned j =0; j < in_nodes1D-2; ++j) {
      nodesToAdd[j] = nodeNumber;
      ++nodeNumber;
    }
    faceNodeMap[faceOrdinal].resize(in_nodes1D);
    faceNodeMap[faceOrdinal][0] = baseEdge[0];
    faceNodeMap[faceOrdinal][in_nodes1D-1] = baseEdge[1];

    for (unsigned j = 1; j < in_nodes1D-1; ++j) {
      faceNodeMap[faceOrdinal][j] = nodesToAdd[j-1];
    }
    edgeNodeConnectivities.insert({nodesToAdd,baseEdge});
    ++faceOrdinal;
  }

  unsigned faceNodeNumber = nodeNumber;
  unsigned nodesLeft = (in_nodes1D*in_nodes1D) - faceNodeNumber;
  std::vector<size_t> faceNodesToAdd(nodesLeft);
  for (unsigned j = 0; j < nodesLeft;++j) {
    faceNodesToAdd[j] = faceNodeNumber;
    ++faceNodeNumber;
  }
  faceNodeConnectivities.insert({faceNodesToAdd,baseFaceNodes});

  for (const auto& edgeNode : edgeNodeConnectivities) {
    addedConnectivities.insert(edgeNode);
  }

  for (const auto& faceNode : faceNodeConnectivities) {
    addedConnectivities.insert(faceNode);
  }
}
//--------------------------------------------------------------------------
void
QuadMElementDescription::set_node_locations(
  const AddedConnectivityOrdinalMap& in_edgeNodeConnectivities,
  const AddedConnectivityOrdinalMap& in_faceNodeConnectivities,
  const std::vector<double>& in_nodeLocs)
{
  unsigned polyOrder = in_nodeLocs.size()-1;
  for (const auto& edgeNode : in_edgeNodeConnectivities) {
    const auto& newNodes = edgeNode.first;
    std::vector<std::vector<double>> locs(polyOrder-1);
    for (unsigned i = 0; i < polyOrder-1; ++i) {
      locs[i].push_back(in_nodeLocs[1+i]);
    }
    locationsForNewNodes.insert({newNodes,locs});
  }

  for (const auto& faceNode : in_faceNodeConnectivities) {
    const auto& newNodes = faceNode.first;
    std::vector<std::vector<double>> locs((polyOrder-1)*(polyOrder-1));
    for (unsigned j = 0; j < polyOrder-1; ++j) {
      for (unsigned i = 0; i < polyOrder-1; ++i) {
        locs[i+(polyOrder-1)*j] = {in_nodeLocs[i+1],in_nodeLocs[j+1]};
      }
    }
    locationsForNewNodes.insert({newNodes, locs});
  }
}
//--------------------------------------------------------------------------
void
QuadMElementDescription::set_subelement_connectivity(unsigned in_nodes1D)
{
  subElementConnectivity.resize((in_nodes1D-1)*(in_nodes1D-1));
  for (unsigned j = 0; j < in_nodes1D-1; ++j) {
    for (unsigned i = 0; i < in_nodes1D-1; ++i) {
      subElementConnectivity[i+(in_nodes1D-1)*j] =
      {
          (size_t)tensor_product_node_map(i,j),
          (size_t)tensor_product_node_map(i+1,j),
          (size_t)tensor_product_node_map(i+1,j+1),
          (size_t)tensor_product_node_map(i,j+1)
      };
    }
  }
}
//--------------------------------------------------------------------------
Hex27ElementDescription::Hex27ElementDescription() :  ElementDescription()
{
  polyOrder = 2;
  nodes1D = polyOrder+1;
  nodesPerElement = nodes1D*nodes1D*nodes1D;
  dimension = 3;

  edgeNodeConnectivities = {
      { {8},  {0, 1}},
      { {9},  {1, 2}},
      { {10}, {2, 3}},
      { {11}, {3, 0}},
      { {12}, {0, 4}},
      { {13}, {1, 5}},
      { {14}, {2, 6}},
      { {15}, {3, 7}},
      { {16}, {4, 5}},
      { {17}, {5, 6}},
      { {18}, {6, 7}},
      { {19}, {7, 4}},
  };

  volumeNodeConnectivities = {
      { {20}, {0, 1, 2, 3, 4, 5, 6, 7}}
  };

  faceNodeConnectivities = {
      { {21}, {0, 3, 2, 1}},
      { {22}, {4, 5, 6, 7}},
      { {23}, {0, 4, 7, 3}},
      { {24}, {1, 2, 6, 5}},
      { {25}, {0, 1, 5, 4}},
      { {26}, {2, 3, 7, 6}}
  };

  for (const auto& edgeNode : edgeNodeConnectivities) {
    addedConnectivities.insert(edgeNode);
  }

  for (const auto& volumeNode : volumeNodeConnectivities) {
    addedConnectivities.insert(volumeNode);
  }

  for (const auto& faceNode : faceNodeConnectivities) {
    addedConnectivities.insert(faceNode);
  }

  locationsForNewNodes = {
      { {8}, {{0.0}}},
      { {9}, {{0.0}}},
      {{10}, {{0.0}}},
      {{11}, {{0.0}}},
      {{12}, {{0.0}}},
      {{13}, {{0.0}}},
      {{14}, {{0.0}}},
      {{15}, {{0.0}}},
      {{16}, {{0.0}}},
      {{17}, {{0.0}}},
      {{18}, {{0.0}}},
      {{19}, {{0.0}}},
      {{20}, {{0.0, 0.0, 0.0}}},
      {{21}, {{0.0, 0.0}}},
      {{22}, {{0.0, 0.0}}},
      {{23}, {{0.0, 0.0}}},
      {{24}, {{0.0, 0.0}}},
      {{25}, {{0.0, 0.0}}},
      {{26}, {{0.0, 0.0}}}
  };

  subElementConnectivity = {
      { 0, 8,21,11,12,25,20,23},
      { 8, 1, 9,21,25,13,24,20},
      { 9, 2,10,21,24,14,26,20},
      {10, 3,11,21,26,15,23,20},
      {12,25,20,23, 4,16,22,19},
      {25,13,24,20,16, 5,17,22},
      {24,14,26,20,17, 6,18,22},
      {26,15,23,20,18, 7,19,22}
  };

}


} // namespace naluUnit
}  // namespace sierra

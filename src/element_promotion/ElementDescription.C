#include <element_promotion/ElementDescription.h>
#include <nalu_make_unique.h>

// stk_mesh
#include <stk_mesh/base/FieldBase.hpp>
#include <utility>

namespace sierra {
namespace naluUnit {

std::unique_ptr<ElementDescription>
ElementDescription::create(std::string type)
{
  if (type == "Quad9") {
    return make_unique<Quad9ElementDescription>();
  }
  if (type == "Quad16") {
    return make_unique<Quad16ElementDescription>();
  }
  if (type == "Hex27") {
    return make_unique<Hex27ElementDescription>();
  }
  throw std::runtime_error("Element type not implemented");
  return nullptr;
}

Quad9ElementDescription::Quad9ElementDescription()
  : ElementDescription()
{
  polyOrder = 2;
  nodes1D = polyOrder + 1;
  nodesPerElement = nodes1D*nodes1D;
  dimension = 2;
  numQuad = 2;
  useGLLGLL = false;

  nodeMap = {
              0, 4, 1, // bottom row of nodes
              7, 8, 5, // middle row of nodes
              3, 6, 2  // top row of nodes
            };

  nodeMap1D = { 0, 2, 1 };

  inverseNodeMap.resize(nodesPerElement);
  for (unsigned i = 0; i < nodes1D; ++i) {
    for (unsigned j = 0; j < nodes1D; ++j) {
      inverseNodeMap[tensor_product_node_map(i,j)] = {i, j};
    }
  }

  inverseNodeMap1D = { {0,0}, {2,1}, {1,2} };

  nodeLocs = { -1.0, 0.0, +1.0 };
  scsLoc = { -std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };

  quadrature = make_unique<TensorProductQuadratureRule>("GaussLegendre", polyOrder, scsLoc);
  basis = make_unique<LagrangeBasis>(inverseNodeMap, nodeLocs);

  unsigned baseNodesPerElement = 4;
  unsigned nodeNumber = baseNodesPerElement;
  std::vector<std::vector<size_t>> baseEdgeNodes = {{0,1},{1,2},{2,3},{3,0}};
  std::vector<size_t> baseFaceNodes = {0,1,2,3};

  for (auto baseEdge : baseEdgeNodes) {
    std::vector<size_t> nodesToAdd(nodes1D-2);
    for (unsigned j =0; j < nodes1D-2; ++j) {
      nodesToAdd[j] = nodeNumber;
      ++nodeNumber;
    }
    edgeNodeConnectivities.insert({nodesToAdd,baseEdge});
  }

  unsigned faceNodeNumber = nodeNumber;
  unsigned nodesLeft = nodesPerElement - faceNodeNumber;
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

  locationsForNewNodes = {
      { { 4}, { {nodeLocs[1]}}},
      { { 5}, { {nodeLocs[1]}}},
      { { 6}, { {nodeLocs[1]}}},
      { { 7}, { {nodeLocs[1]}}},
      { { 8}, { {nodeLocs[1], nodeLocs[1]}}},
  };

  subElementConnectivity = { {4,0,7,8}, {4,1,5,8}, {6,2,5,8}, {6,3,7,8} };
};
//--------------------------------------------------------------------------
Quad16ElementDescription::Quad16ElementDescription() : ElementDescription()
{
  polyOrder = 3;
  nodes1D = polyOrder + 1;
  nodesPerElement = nodes1D * nodes1D;
  dimension = 2;
  numQuad = 2;
  useGLLGLL = false;

  nodeMap =
  {
      0, 4, 5, 1,
      11, 12, 13, 6,
      10, 14, 15, 7,
      3, 9, 8, 2
  };

  nodeMap1D = { 0, 2, 3, 1 };

  inverseNodeMap.resize(nodesPerElement);
  for (unsigned i = 0; i < nodes1D; ++i) {
    for (unsigned j = 0; j < nodes1D; ++j) {
      inverseNodeMap[tensor_product_node_map(i,j)] = {i, j};
    }
  }

  inverseNodeMap1D = { {0,0}, {2,1}, {3,2}, {1,3} };

  //symmetrizing points for the mass matrix
  double xgll = 0.4487053820572093009546164613323186035;
  double scsDist = 0.8347278713337825805263131558586123084;
  nodeLocs = { -1.0, -xgll, +xgll, +1.0 };
  scsLoc = { -scsDist, 0.0, scsDist };

  quadrature = make_unique<TensorProductQuadratureRule>("GaussLegendre", polyOrder, scsLoc);
  basis = make_unique<LagrangeBasis>(inverseNodeMap, nodeLocs);


  unsigned baseNodesPerElement = 4;
  unsigned nodeNumber = baseNodesPerElement;
  std::vector<std::vector<size_t>> baseEdgeNodes = {{0,1},{1,2},{2,3},{3,0}};
  std::vector<size_t> baseFaceNodes = {0,1,2,3};

  for (auto baseEdge : baseEdgeNodes) {
    std::vector<size_t> nodesToAdd(nodes1D-2);
    for (unsigned j =0; j < nodes1D-2; ++j) {
      nodesToAdd[j] = nodeNumber;
      ++nodeNumber;
    }
    edgeNodeConnectivities.insert({nodesToAdd,baseEdge});
  }

  unsigned faceNodeNumber = nodeNumber;
  unsigned nodesLeft = nodesPerElement - faceNodeNumber;
  std::vector<size_t> faceNodesToAdd(nodesLeft);
  for (unsigned j = 0; j < nodesLeft;++j) {
    faceNodesToAdd[j] = faceNodeNumber;
    ++faceNodeNumber;
  }
  faceNodeConnectivities.insert({faceNodesToAdd,baseFaceNodes});

  for (const auto& edgeNode : edgeNodeConnectivities) {
    addedConnectivities.insert(edgeNode);
    const auto& newNodes = edgeNode.first;
    std::vector<std::vector<double>> locsForNewNode(polyOrder-1);
    for (unsigned i = 0; i < polyOrder-1; ++i) {
      locsForNewNode[i].push_back(nodeLocs[1+i]);
    }
    locationsForNewNodes.insert({newNodes,locsForNewNode});
  }

  for (const auto& faceNode : faceNodeConnectivities) {
    addedConnectivities.insert(faceNode);
  }

  locationsForNewNodes.insert(
       { {12,13,14,15}, { {nodeLocs[1], nodeLocs[1] },
                         { nodeLocs[2], nodeLocs[1] },
                         { nodeLocs[1], nodeLocs[2] },
                         { nodeLocs[2], nodeLocs[2] } } }
  );


  subElementConnectivity = {
      { 0, 4,12,11}, { 4, 5,13,12}, { 5, 1, 6,13},
      { 6, 7,15,13}, { 7, 2, 8,15}, { 8, 9,14,15},
      { 9, 3,10,14}, {10,11,12,14}, {12,13,15,14}
  };
};
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



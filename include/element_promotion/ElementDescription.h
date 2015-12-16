#ifndef ElementDescription_h
#define ElementDescription_h

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

// STL
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <array>
#include <string>

namespace sierra {
namespace naluUnit {

typedef std::map<std::vector<size_t>, std::vector<size_t>> AddedConnectivityOrdinalMap;
typedef std::map<std::vector<size_t>, std::vector<std::vector<double>>> AddedNodeLocationsMap;
typedef std::vector<std::vector<size_t>> SubElementConnectivity;

struct ElementDescription
{
public:
  static std::unique_ptr<ElementDescription> create(std::string type);
  virtual ~ElementDescription() {};

  inline int tensor_product_node_map(int i, int j, int k) const
  {
    return nodeMap[i+nodes1D*(j+nodes1D*k)];
  }

  inline int tensor_product_node_map(int i, int j) const
  {
    return nodeMap[i+nodes1D*j];
  }

  inline int tensor_product_node_map(int i) const
  {
    return nodeMap1D[i];
  }

  inline double gauss_point_location(
    int nodeOrdinal,
    int gaussPointOrdinal) const
  {
    return quadrature->gauss_point_location(nodeOrdinal,gaussPointOrdinal);
  }

  inline double tensor_product_weight(
    int s1Node, int s2Node,
    int s1Ip, int s2Ip) const
  {
    return quadrature->tensor_product_weight(s1Node,s2Node, s1Ip,s2Ip);
  }

  inline double tensor_product_weight(int s1Node, int s1Ip) const
  {
    return quadrature->tensor_product_weight(s1Node, s1Ip);
  }

  inline std::vector<double>
  eval_basis_weights(unsigned dimension, std::vector<double>& intgLoc) const
  {
     return basis->eval_basis_weights(dimension, intgLoc);
  }

  inline std::vector<double>
  eval_deriv_weights(unsigned dimension, std::vector<double>& intgLoc) const
  {
    return basis->eval_deriv_weights(dimension, intgLoc);
  }

  size_t dimension;
  size_t nodes1D;
  size_t nodesPerElement;
  AddedConnectivityOrdinalMap addedConnectivities;
  AddedConnectivityOrdinalMap edgeNodeConnectivities;
  AddedConnectivityOrdinalMap faceNodeConnectivities;
  AddedConnectivityOrdinalMap volumeNodeConnectivities;
  AddedNodeLocationsMap locationsForNewNodes;
  SubElementConnectivity subElementConnectivity;

  bool useGLLGLL;
  unsigned polyOrder;
  unsigned numQuad;
  std::unique_ptr<TensorProductQuadratureRule> quadrature; // change to unique
  std::unique_ptr<LagrangeBasis> basis;
  std::vector<unsigned> nodeMap;
  std::vector<unsigned> nodeMap1D;
  std::vector<double> scsLoc;
  std::vector<double> nodeLocs;
  std::vector<std::vector<unsigned>> inverseNodeMap;
  std::map<unsigned,unsigned> inverseNodeMap1D;
protected:
  ElementDescription() = default;
};

struct Quad9ElementDescription: public ElementDescription
{
  Quad9ElementDescription();
};

struct Quad16ElementDescription: public ElementDescription
{
  Quad16ElementDescription();
};


struct Hex27ElementDescription: public ElementDescription
{
  Hex27ElementDescription();
};

} // namespace naluUnit
} // namespace Sierra

#endif

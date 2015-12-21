#ifndef ElementDescription_h
#define ElementDescription_h

#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sierra {
namespace naluUnit {

typedef std::map<std::vector<size_t>, std::vector<size_t>> AddedConnectivityOrdinalMap;
typedef std::map<std::vector<size_t>, std::vector<std::vector<double>>> AddedNodeLocationsMap;
typedef std::vector<std::vector<size_t>> SubElementConnectivity;

struct ElementDescription
{
public:
  static std::unique_ptr<ElementDescription> create(std::string type);
  virtual ~ElementDescription() = default;

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
  eval_basis_weights(const std::vector<double>& intgLoc) const
  {
     return basis->eval_basis_weights(intgLoc);
  }

  inline std::vector<double>
  eval_deriv_weights(const std::vector<double>& intgLoc) const
  {
    return basis->eval_deriv_weights(intgLoc);
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
  std::unique_ptr<TensorProductQuadratureRule> quadrature;
  std::unique_ptr<LagrangeBasis> basis;
  std::unique_ptr<LagrangeBasis> basisBoundary;
  std::vector<unsigned> nodeMap;
  std::vector<unsigned> nodeMap1D;
  std::vector<double> scsLoc;
  std::vector<double> nodeLocs;
  std::vector<std::vector<unsigned>> inverseNodeMap;
  std::vector<std::vector<unsigned>> inverseNodeMap1D;
  std::vector<std::vector<size_t>> faceNodeMap;
protected:
  ElementDescription() = default;
};

struct QuadMElementDescription: public ElementDescription
{
  QuadMElementDescription(const std::vector<double> in_nodeLocs, const std::vector<double>& in_scsLoc);

private:
  void set_node_maps(unsigned in_nodes1D);
  void set_node_connectivity(unsigned in_nodes1D);

  void set_node_locations(
    const AddedConnectivityOrdinalMap& in_edgeNodeConnectivities,
    const AddedConnectivityOrdinalMap& in_faceNodeConnectivities,
    const std::vector<double>& in_nodeLocs
  );

  void set_subelement_connectivity(unsigned in_nodes1D);
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

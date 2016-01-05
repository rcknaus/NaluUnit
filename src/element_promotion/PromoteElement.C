/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromoteElement.h>

#include <element_promotion/ElementDescription.h>
#include <element_promotion/FaceOperations.h>
#include <NaluEnv.h>

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_util/parallel/CommSparse.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/BulkDataInlinedMethods.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_topology/topology.hpp>
#include <stk_topology/topology.tcc>
#include <stk_util/environment/ReportHandler.hpp>
#include <stk_util/parallel/ParallelComm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>

namespace sierra{
namespace naluUnit{

  std::string promote_part_name(const std::string& base_name)
  {
    return (base_name+"_promoted");
  }

  stk::mesh::Part* promoted_part(const stk::mesh::Part& part) {
    auto promotedName = promote_part_name(part.name());
    return (part.mesh_meta_data().get_part(promotedName));
  }

//==========================================================================
// Class Definition
//==========================================================================
// PromoteElement - Promotes a mesh based on a description of the new element
// connectivities and node locations
// TODO(rcknaus): allow some parts not to be promoted
//===============================c===========================================
PromoteElement::PromoteElement(ElementDescription& elemDescription)
: elemDescription_(elemDescription),
  nodesPerElement_(elemDescription.nodesPerElement),
  dimension_(elemDescription.dimension)
{
 //do nothing
}
//--------------------------------------------------------------------------
size_t PromoteElement::num_elems(const stk::mesh::Entity& node) const
{
  return (nodeElemMap_.at(node).size());
}
//--------------------------------------------------------------------------
size_t PromoteElement::num_nodes(const stk::mesh::Entity& elem) const
{
  return (elemNodeMap_.at(elem).size());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_nodes_all(const stk::mesh::Bucket& bucket,
  stk::mesh::EntityId id) const
{
  return (elemNodeMap_.at(bucket[id]).data());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_nodes_all(const stk::mesh::Entity& elem) const
{
  return (elemNodeMap_.at(elem).data());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_elems_all(const stk::mesh::Bucket& bucket,
  stk::mesh::EntityId id) const
{
  return (nodeElemMap_.at(bucket[id]).data());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_elems_all(const stk::mesh::Entity& elem) const
{
  return (nodeElemMap_.at(elem).data());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_side_elems_all(const stk::mesh::Bucket& bucket,
  stk::mesh::EntityId id) const
{
  return (nodeElemMapBC_.at(bucket[id]).data());
}
//--------------------------------------------------------------------------
stk::mesh::Entity const*
PromoteElement::begin_side_elems_all(const stk::mesh::Entity& elem) const
{
  return (nodeElemMapBC_.at(elem).data());
}
//--------------------------------------------------------------------------
void
PromoteElement::promote_elements(const stk::mesh::PartVector& baseParts,
  VectorFieldType& coordinates, stk::mesh::BulkData& mesh,
  stk::mesh::PartVector& promotedParts)
{
  ThrowRequireMsg(mesh.in_modifiable_state(),
    "Mesh must be in a modifiable state for element promotion");

  baseParts_ = baseParts; // holds the original elements and original nodes
  promotedParts_ = promotedParts; // holds only the new nodes (

  auto basePartSelector = stk::mesh::selectUnion(baseParts);

  auto timeB = MPI_Wtime();
  auto nodeRequests = create_child_node_requests(mesh, elemDescription_,
    basePartSelector);
  auto timeC = MPI_Wtime();
  NaluEnv::self().naluOutputP0() << "P0: Time to generate requests: "
      << (timeC - timeB) << std::endl;

  batch_create_child_nodes(mesh, nodeRequests, baseParts);
  auto timeD = MPI_Wtime();
  NaluEnv::self().naluOutputP0() << "P0: Time to create requested nodes: "
      << (timeD - timeC) << std::endl;

  populate_elem_node_relations(elemDescription_, mesh, basePartSelector,
    nodeRequests);
  auto timeE = MPI_Wtime();
  NaluEnv::self().naluOutputP0()
              << "P0: Time to save node relations to map: " << (timeE - timeD)
              << std::endl;

  if (dimension_ == 2) {
    set_new_node_coords<2>(coordinates, elemDescription_, nodeRequests);
  }
  else {
    set_new_node_coords<3>(coordinates, elemDescription_, nodeRequests);
  }
  auto timeF = MPI_Wtime();
  NaluEnv::self().naluOutputP0() << "P0: Time to set node coordinates: "
      << (timeF - timeE) << std::endl;
}
//--------------------------------------------------------------------------
PromoteElement::NodeRequests
PromoteElement::create_child_node_requests(
  stk::mesh::BulkData& mesh,
  const ElementDescription& elemDescription,
  const stk::mesh::Selector& selector) const
{
  // Creates a list of nodes to be created by the batch_create_child_nodes method.
  // Saves off a list of elements sharing the added nodes as well as the ordinal
  // the new node should have for each element it is associated with
  const stk::mesh::BucketVector& elem_buckets = mesh.get_buckets(
    stk::topology::ELEM_RANK, selector);

  const auto& connectivities = elemDescription.addedConnectivities;
  const size_t num_relations = connectivities.size();

  // pre-size parentIds
  std::vector<std::vector<stk::mesh::EntityId>> parentIds(num_relations);
  size_t countRels = 0;
  for (const auto& relation : connectivities) {
    parentIds[countRels].resize(relation.second.size());
    ++countRels;
  }

  NodeRequests requestSet;
  for (const auto* ib : elem_buckets) {
    const stk::mesh::Bucket& b = *ib;
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const auto& elem = b[k];
      const stk::mesh::Entity* nodes = b.begin_nodes(k);
      size_t relationCount = 0;
      for (const auto& relation : connectivities) {
        auto& parentOrdinals = relation.second;
        for (size_t j = 0; j < parentOrdinals.size(); ++j) {
          parentIds[relationCount][j] = mesh.identifier(nodes[parentOrdinals[j]]);
        }

        auto unsortedParentIds = parentIds[relationCount];
        std::sort(
          parentIds[relationCount].begin(),
          parentIds[relationCount].end()
        );

        auto result =
            requestSet.insert(ChildNodeRequest{ parentIds[relationCount] });
        result.first->add_shared_elem(elem); // add a shared elem regardless
        if (result.second) {
          result.first->set_num_children(relation.first.size());
        }
        result.first->unsortedParentIds_ = unsortedParentIds;
        ++relationCount;
      }
    }
  }

  //FIXME(rcknaus): clean-up the ordinal reversing logic

  // For P>2, we have to worry about whether an edge is ordered forward or backward
  // The locations map expects things to be ordered, so we keep them sorted
  // and the node locations are reversed in elemNodeMap.
  unsigned nodes1D = elemDescription.nodes1D;
  unsigned numAddedNodes1D = nodes1D-2;
  unsigned numParents1D = nodes1D-numAddedNodes1D; //2
  for (auto& request : requestSet) {
    unsigned numShared = request.sharedElems_.size();
    request.childOrdinalsForElem_.resize(numShared);
    request.reorderedChildOrdinalsForElem_.resize(numShared);
    request.unsortedParentOrdinals_.resize(numShared);
    request.reverseSharing_ = false;
    for (unsigned elemNumber = 0; elemNumber < numShared; ++elemNumber) {
      auto ordinals = request.determine_child_node_ordinal(mesh, elemDescription, elemNumber);
      auto& canonicalOrdinals = elemDescription.addedConnectivities.at(ordinals);
      auto& unsortedOrdinals = request.unsortedParentOrdinals_[elemNumber];

      if (parents_are_reversed<size_t>(unsortedOrdinals, canonicalOrdinals)) {
        std::reverse(ordinals.begin(), ordinals.end());
        if(unsortedOrdinals.size() == numParents1D*numParents1D) {
          flip_x<size_t>(ordinals,numAddedNodes1D);
        }
      }
      else if (parents_are_flipped_x<size_t>(unsortedOrdinals, canonicalOrdinals, numParents1D)) {
        flip_x<size_t>(ordinals,numAddedNodes1D);
      }
      else if (parents_are_flipped_y<size_t>(unsortedOrdinals, canonicalOrdinals, numParents1D)) {
        flip_y<size_t>(ordinals,numAddedNodes1D); // never executed AFAIK
      }
      else if (unsortedOrdinals != canonicalOrdinals) {
        transpose_ordinals<size_t>(ordinals,numAddedNodes1D);
        // if transposed, the "off-diagional" nodes (i.e. (1,0)->1 and (0,1)->3) should be inverted
        if (unsortedOrdinals[1] != canonicalOrdinals[3]) {
          throw std::runtime_error("Element promotion: unexpected permutation of parent ordinals");
        }
      }
      else {
        // swap sharing in the opposite case for when parents are reversed

        // FIXME(rcknaus): in 3D, this should be an enum denoting
        // the correct ordinal transformation to apply (e.g. transpose_ordinals)
        if (dimension_ == 2) {
          request.reverseSharing_ = true;
        }
      }
      request.reorderedChildOrdinalsForElem_[elemNumber] = std::move(ordinals);
    }
  }


  return requestSet;
}
//--------------------------------------------------------------------------
void
PromoteElement::batch_create_child_nodes(
  stk::mesh::BulkData& mesh,
  NodeRequests& requests,
  const stk::mesh::PartVector& node_parts) const
{
  size_t num_nodes_requested = count_requested_nodes(requests);
  std::vector<stk::mesh::EntityId> available_node_ids(num_nodes_requested);
  mesh.generate_new_ids(stk::topology::NODE_RANK, num_nodes_requested,
    available_node_ids);

  size_t it_req = 0;
  for (auto& request : requests) {
    for (unsigned j = 0; j < request.num_children(); ++j, ++it_req) {
      request.add_proc_id_pair(
        mesh.parallel_rank(), available_node_ids[it_req], j
      );
    }
    request.determine_sharing_procs(mesh);
  }

  if (mesh.parallel_size() > 1) {
    parallel_communicate_ids(mesh, requests);
  }


  for (auto& request : requests) {
     request.set_node_entity_for_request(mesh, node_parts);
  }
}
//--------------------------------------------------------------------------
void
PromoteElement::parallel_communicate_ids(
  const stk::mesh::BulkData& mesh, NodeRequests& requests) const
{
  stk::CommSparse comm_spec(mesh.parallel());

  for (int phase = 0; phase < 2; ++phase) {
    for (const auto& request : requests) {
      for (auto other_proc : request.sharingProcs_) {
        if (other_proc != mesh.parallel_rank()) {
          const auto& request_parents = request.parentIds_;
          const auto numChildren = request.num_children();
          comm_spec.send_buffer(other_proc).pack(request.num_parents());
          comm_spec.send_buffer(other_proc).pack(numChildren);
          for (auto parentId : request_parents) {
            comm_spec.send_buffer(other_proc).pack(parentId);
          }

          for (unsigned j = 0; j < numChildren; ++j) {
            comm_spec.send_buffer(other_proc).pack(
              request.suggested_node_id(j));
          }
        }
      }
    }

    if (phase == 0) {
      comm_spec.allocate_buffers();
    }
    else {
      comm_spec.communicate();
    }
  }

  for (int i = 0; i < mesh.parallel_size(); ++i) {
    if (i != mesh.parallel_rank()) {
      while (comm_spec.recv_buffer(i).remaining() != 0) {
        size_t num_parents;
        size_t num_children;
        stk::mesh::EntityId suggested_node_id;
        comm_spec.recv_buffer(i).unpack(num_parents);
        comm_spec.recv_buffer(i).unpack(num_children);
        std::vector<stk::mesh::EntityId> parentIds(num_parents);
        for (auto& parentId : parentIds) {
          comm_spec.recv_buffer(i).unpack(parentId);
        }
        auto iter = requests.find(ChildNodeRequest{ parentIds });

        bool hasParents = iter != requests.end();
        for (unsigned j = 0; j < num_children; ++j) {
          comm_spec.recv_buffer(i).unpack(suggested_node_id);
          if (hasParents) {
            int jr = j;
            if (iter->reverseSharing_) {
              jr = iter->childOrdinalsForElem_[0].size()-j-1;
            }
            iter->add_proc_id_pair(i, suggested_node_id, jr);
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
void
PromoteElement::populate_elem_node_relations(
  const ElementDescription& elemDescription, stk::mesh::BulkData& mesh,
  const stk::mesh::Selector selector, const NodeRequests& requests)
{
  const stk::mesh::BucketVector& elem_buckets = mesh.get_buckets(
    stk::topology::ELEM_RANK, selector);
  const stk::mesh::BucketVector& node_buckets = mesh.get_buckets(
    stk::topology::NODE_RANK, selector);

  elemNodeMap_.reserve(count_entities(elem_buckets));
  nodeElemMap_.reserve(count_requested_nodes(requests));

  // initialize base downward relationships
  for (const auto* ib : elem_buckets) {
    const stk::mesh::Bucket& b = *ib;
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const stk::mesh::Entity elem = b[k];
      const stk::mesh::Entity* nodes = b.begin_nodes(k);
      elemNodeMap_[elem].resize(nodesPerElement_);
      for (size_t j = 0; j < b.num_nodes(k); ++j) {
        elemNodeMap_[elem][j] = nodes[j];
      }
    }
  }

  // initialize base upward relationships
  for (const auto* ib : node_buckets) {
    const stk::mesh::Bucket& b = *ib;
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const stk::mesh::Entity node = b[k];
      const stk::mesh::Entity* elem_rels = b.begin_elements(k);
      const size_t num_elems = b.num_elements(k);
      std::vector<stk::mesh::Entity> elemList(num_elems);
      nodeElemMap_.insert({ node, elemList });
      for (size_t j = 0; j < b.num_elements(k); ++j) {
        nodeElemMap_.at(node)[j] = elem_rels[j];
      }
    }
  }

  unsigned nodes1D = elemDescription.nodes1D;
  for (const auto& request : requests) {
    unsigned numShared = request.sharedElems_.size();
    request.childOrdinalsForElem_.resize(numShared);
    request.unsortedParentOrdinals_.resize(numShared);

    request.elemIndex_ = 0;
    for (unsigned j = 0; j < numShared; ++j) {
      auto& ordinals = request.reorderedChildOrdinalsForElem_[j];
      auto& childOrdinals = request.childOrdinalsForElem_[j];
      if (ordinals == childOrdinals) {
        request.elemIndex_ = j;
        break;
      }
    }

    for (unsigned elemNumber = 0; elemNumber < numShared; ++elemNumber) {
      auto sharedElem = request.sharedElems_[elemNumber];
      auto& ordinals = request.reorderedChildOrdinalsForElem_[elemNumber];

      for (unsigned j = 0; j < request.num_children(); ++j) {
        elemNodeMap_.at(sharedElem)[ordinals[j]] = request.children_[j];
      }
    }
    for (const auto child : request.children_) {
      nodeElemMap_.insert({ child, request.sharedElems_ });
    }
  }

  // Boundaries
    auto topo = (dimension_ == 2) ?
        stk::topology::EDGE_RANK : stk::topology::FACE_RANK;
    const stk::mesh::BucketVector& boundary_buckets = mesh.get_buckets(
      topo, selector);

    //elemNodeMapBC_.reserve(count_entities(boundary_buckets));
    nodeElemMapBC_.reserve(
      count_entities(boundary_buckets)*elemDescription_.nodes1D
    );

    std::vector<stk::mesh::Entity> faceNodes(std::pow(nodes1D,dimension_-1));
    for (const auto* ib : boundary_buckets) {
      const stk::mesh::Bucket& b = *ib;
      const stk::mesh::Bucket::size_type length = b.size();

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        const auto face = b[k];

        const auto* parent_elems = mesh.begin_elements(face);
        ThrowAssert(mesh.num_elements(face) == 1);

        const auto* face_elem_ords = mesh.begin_element_ordinals(face);
        const unsigned face_ordinal = face_elem_ords[0];

        const auto elem_node_rels = elemNodeMap_.at(parent_elems[0]);

        auto nodeOrdinalsForFace = elemDescription_.faceNodeMap[face_ordinal];

        if (dimension_ == 2) {
          for (unsigned j = 0; j < nodes1D; ++j) {
            int ordinal;
            if (face_ordinal == 2 || face_ordinal ==3) {
              ordinal = elemDescription_.tensor_product_node_map(nodes1D-j-1);
            }
            else {
              ordinal = elemDescription_.tensor_product_node_map(j);
            }
            faceNodes[ordinal] = elem_node_rels.at(nodeOrdinalsForFace.at(j));
          }
        }
        else {
          for (unsigned j = 0; j < nodes1D; ++j) {
            for (unsigned i = 0; i < nodes1D; ++i) {
              int ordinal;
              if (face_ordinal == 2) {
                ordinal = elemDescription_.tensor_product_node_map_bc(nodes1D-i-1,j);
               }
              else if (face_ordinal == 3 || face_ordinal == 4) {
                ordinal = elemDescription_.tensor_product_node_map_bc(j,i);
              }
              else {
                ordinal = elemDescription_.tensor_product_node_map_bc(i,j);
              }
              faceNodes[ordinal] = elem_node_rels.at(
                nodeOrdinalsForFace.at(i+nodes1D*j)
              );
            }
          }
        }

        elemNodeMap_.insert({face, faceNodes});

        for (auto faceNode : faceNodes) {
          nodeElemMapBC_.insert({faceNode, {face}});
        }
      }
    }

  ThrowAssert(check_elem_node_relations(mesh));
}
//--------------------------------------------------------------------------
bool
PromoteElement::check_elem_node_relations(
  const stk::mesh::BulkData& mesh) const
{
  for (const auto& elemNodePair : elemNodeMap_) {
    for (const auto& node : elemNodePair.second) {
      if (!(mesh.is_valid(node))) {
        return false;
      }
    }
  }

  for (const auto& nodeElemPair : nodeElemMap_) {
    if (nodeElemPair.second.size() < 1) {
      return false;
    }
    for (const auto& elem : nodeElemPair.second) {
      if (!(mesh.is_valid(elem))) {
        return false;
      }
    }
  }
  return true;
}
//--------------------------------------------------------------------------
template<unsigned embedding_dimension> void
PromoteElement::set_new_node_coords(
  VectorFieldType& coordinates,
  const ElementDescription& elemDescription,
   NodeRequests& requests) const
{
  //  hex/quad specific method for interpolating coordinates
  static_assert(embedding_dimension == 2 || embedding_dimension == 3,"");

  for (auto& request : requests) {
    unsigned elemIndex = 0u;

    auto numParents = request.parentIds_.size();
    auto& ordinals= request.childOrdinalsForElem_[elemIndex];
    auto* node_rels = begin_nodes_all(request.sharedElems_[elemIndex]);
    auto& parentOrdinals = elemDescription.addedConnectivities.at(ordinals);
    auto& childLocations = elemDescription.locationsForNewNodes.at(ordinals);

    //auto parentOrdinals = request.unsortedParentOrdinals_[elemIndex];

    switch (numParents)
    {
      case 2:
      {
        set_coords_for_child<embedding_dimension, 1>(
          coordinates, node_rels,
          ordinals, parentOrdinals,
          childLocations);
        break;
      }
      case 4:
      {
        set_coords_for_child<embedding_dimension, 2>(
          coordinates, node_rels,
          ordinals, parentOrdinals,
          childLocations);
        break;
      }
      case 8:
      {
        set_coords_for_child<3, 3>(
          coordinates, node_rels,
          ordinals, parentOrdinals,
          childLocations);
        break;
      }
      default:
      {
        throw std::runtime_error("invalid parent number");
      }
    }
  }
}
//--------------------------------------------------------------------------
template<unsigned embedding_dimension, unsigned dimension> void
PromoteElement::set_coords_for_child(
  VectorFieldType& coordinates, const stk::mesh::Entity* node_rels,
  std::vector<size_t>& childOrdinal,
  const std::vector<size_t>& parentNodeOrdinals,
  const std::vector<std::vector<double>>& isoParCoords) const
{
  constexpr unsigned numParents = ipow(2,dimension);
  ThrowAssert(parentNodeOrdinals.size() == numParents);

  std::array<double*, numParents> parentCoordPtrs;
  for (unsigned m = 0; m < numParents; ++m) {
    parentCoordPtrs[m] = static_cast<double*>(stk::mesh::field_data(
      coordinates, node_rels[parentNodeOrdinals[m]]
    ));
  }

  std::array<double, embedding_dimension * numParents> parentCoords;
  for (size_t m = 0; m < numParents; ++m) {
    for (size_t j = 0; j < embedding_dimension; ++j) {
      parentCoords[j + m * embedding_dimension] = parentCoordPtrs[m][j];
    }
  }

  for (unsigned j = 0; j < childOrdinal.size(); ++j) {
    auto* coords =
        static_cast<double*>(
            stk::mesh::field_data(coordinates, node_rels[childOrdinal[j]]
        )
    );

    interpolate_coords<embedding_dimension, dimension>(
      isoParCoords[j].data(),
      parentCoords,
      coords
    );
  }
}
//--------------------------------------------------------------------------
template<unsigned embedding_dimension, unsigned dimension> void
PromoteElement::interpolate_coords(
  const double* isoParCoord,
  const std::array<double, embedding_dimension*ipow(2,dimension)>& parentCoords,
  double* interpolatedCoords) const
{
  static_assert ( embedding_dimension == 2 || embedding_dimension == 3, "");
  static_assert ( dimension <= embedding_dimension, "");

  constexpr unsigned num_shape = ipow(2,dimension);
  std::array<double, num_shape> shape_function;

  auto shape1D = [](double x, double xi) { return 0.5*(1.0+xi*x); };
  switch (dimension) {
    case 1:
    {
      shape_function[0] = shape1D(isoParCoord[0],-1.0);
      shape_function[1] = shape1D(isoParCoord[0],+1.0);
      break;
    }
    case 2:
    {
      const double s1 = isoParCoord[0];
      const double s2 = isoParCoord[1];
      shape_function[0] = shape1D(s1,-1.0)*shape1D(s2,-1.0);
      shape_function[1] = shape1D(s1,+1.0)*shape1D(s2,-1.0);
      shape_function[2] = shape1D(s1,+1.0)*shape1D(s2,+1.0);
      shape_function[3] = shape1D(s1,-1.0)*shape1D(s2,+1.0);
      break;
    }
    case 3:
    {
      const double s1 = isoParCoord[0];
      const double s2 = isoParCoord[1];
      const double s3 = isoParCoord[2];
      shape_function[0] = shape1D(s1,-1.0)*shape1D(s2,-1.0)*shape1D(s3,-1.0);
      shape_function[1] = shape1D(s1,+1.0)*shape1D(s2,-1.0)*shape1D(s3,-1.0);
      shape_function[2] = shape1D(s1,+1.0)*shape1D(s2,-1.0)*shape1D(s3,+1.0);
      shape_function[3] = shape1D(s1,-1.0)*shape1D(s2,-1.0)*shape1D(s3,+1.0);
      shape_function[4] = shape1D(s1,-1.0)*shape1D(s2,+1.0)*shape1D(s3,-1.0);
      shape_function[5] = shape1D(s1,+1.0)*shape1D(s2,+1.0)*shape1D(s3,-1.0);
      shape_function[6] = shape1D(s1,+1.0)*shape1D(s2,+1.0)*shape1D(s3,+1.0);
      shape_function[7] = shape1D(s1,-1.0)*shape1D(s2,+1.0)*shape1D(s3,+1.0);
      break;
    }
  }

  for (unsigned j = 0; j < embedding_dimension; ++j) {
    interpolatedCoords[j] = 0.0;
  }

  for (unsigned m = 0; m < num_shape; ++m) {
    for (unsigned j = 0; j < embedding_dimension; ++j) {
      interpolatedCoords[j] += shape_function[m] *
          parentCoords[j + m * embedding_dimension];
    }
  }
}
//--------------------------------------------------------------------------
size_t
PromoteElement::count_requested_nodes(const NodeRequests& requests) const
{
  size_t numNodes = 0;
  for (const auto& request : requests) {
    numNodes += request.num_children();
  }
  return numNodes;
}
//--------------------------------------------------------------------------
size_t
PromoteElement::num_sub_elements(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BucketVector& buckets) const
{
  unsigned numEntities = 0;
  for (const auto* ib : buckets) {
    unsigned subElemsPerElem =
        (ib->topology().rank() == metaData.side_rank()) ?
            std::pow(elemDescription_.polyOrder, dimension_ - 1) :
            std::pow(elemDescription_.polyOrder, dimension_);

    numEntities += ib->size()*subElemsPerElem;
  }
  return (numEntities);
}
//--------------------------------------------------------------------------
size_t
PromoteElement::count_entities(const stk::mesh::BucketVector& buckets) const
{
  unsigned numEntities = 0;
  for (const auto* ib : buckets) {
    numEntities += ib->size();
  }
  return numEntities;
}
//==========================================================================
// Class Definition
//==========================================================================
// ChildNodeRequest - Provides some utilities to help promote elements
//==========================================================================
PromoteElement::ChildNodeRequest::ChildNodeRequest(
  std::vector<stk::mesh::EntityId>  in_parentIds)
: parentIds_(std::move(in_parentIds))
{
  ThrowAssert(std::is_sorted(parentIds_.begin(), parentIds_.end()));
}
//--------------------------------------------------------------------------
void
PromoteElement::ChildNodeRequest::set_node_entity_for_request(
  stk::mesh::BulkData& mesh,
  const stk::mesh::PartVector& base_parts) const
{
  auto parentParts = determine_parent_parts(mesh, base_parts);
  ThrowRequire(!parentParts.empty());

  stk::mesh::PartVector childParts(parentParts.size());
  for (unsigned j = 0; j < parentParts.size(); ++j) {
    const stk::mesh::Part& parentPart = *(parentParts[j]);
    auto* part = promoted_part(parentPart);;
    ThrowRequire(part != nullptr);

    childParts[j] = part;
  }

  for (unsigned j = 0; j < children_.size(); ++j) {
    auto& idProcPairs = idProcPairsFromAllProcs_[j];
    std::sort(idProcPairs.begin(), idProcPairs.end());

    children_[j] = mesh.declare_entity(
      stk::topology::NODE_RANK, get_id_for_child(j), childParts
    );

    for (auto& idProcPair : idProcPairs) {
      if (idProcPair.first != mesh.parallel_rank()) {
        mesh.add_node_sharing(children_[j], idProcPair.first);
      }
    }
  }
}
//--------------------------------------------------------------------------
stk::mesh::PartVector
PromoteElement::ChildNodeRequest::determine_parent_parts(
  const stk::mesh::BulkData& mesh,
  stk::mesh::PartVector base_parts) const
{
  std::sort(base_parts.begin(), base_parts.end());

  for (unsigned i = 0; i < parentIds_.size(); ++i) {
    auto parentNode = mesh.get_entity(stk::topology::NODE_RANK, parentIds_[i]);
    stk::mesh::PartVector otherParts = mesh.bucket(parentNode).supersets();
    std::sort(otherParts.begin(), otherParts.end());

    stk::mesh::PartVector temp;
    std::set_intersection(
      base_parts.begin(), base_parts.end(),
      otherParts.begin(), otherParts.end(),
      std::back_inserter(temp)
    );
    base_parts = std::move(temp);
  }
  return base_parts;
}
//--------------------------------------------------------------------------
void
PromoteElement::ChildNodeRequest::determine_sharing_procs(
  const stk::mesh::BulkData& mesh) const
{
  ThrowAssert(!parentIds_.empty());

  mesh.comm_shared_procs(
    { stk::topology::NODE_RANK, parentIds_[0] }, sharingProcs_
  );
  ThrowAssert(std::is_sorted(sharingProcs_.begin(), sharingProcs_.end()));

  std::vector<int> parentSharingProcs;
  for (unsigned i = 1; i < parentIds_.size(); ++i) {
    mesh.comm_shared_procs(
      { stk::topology::NODE_RANK, parentIds_[i] }, parentSharingProcs
    );
    ThrowAssert(std::is_sorted(parentSharingProcs.begin(), parentSharingProcs.end()));

    std::vector<int> temp;
    std::set_intersection(
      sharingProcs_.begin(), sharingProcs_.end(),
      parentSharingProcs.begin(), parentSharingProcs.end(),
      std::back_inserter(temp)
    );
    sharingProcs_ = std::move(temp);
  }
}
//--------------------------------------------------------------------------
void
PromoteElement::ChildNodeRequest::add_proc_id_pair(
  int proc_id,
  stk::mesh::EntityId id,
  int childNumber) const
{
   idProcPairsFromAllProcs_[childNumber].emplace_back(proc_id, id);
}
//--------------------------------------------------------------------------
void
PromoteElement::ChildNodeRequest::add_shared_elem(
  const stk::mesh::Entity& elem) const
{
  sharedElems_.push_back(elem);
}
//--------------------------------------------------------------------------
stk::mesh::EntityId
PromoteElement::ChildNodeRequest::get_id_for_child(int childNumber) const
{
  ThrowAssert(std::is_sorted(
    idProcPairsFromAllProcs_[childNumber].begin(),
    idProcPairsFromAllProcs_[childNumber].end()
  ));
  return idProcPairsFromAllProcs_[childNumber][0].second;
}
//--------------------------------------------------------------------------
stk::mesh::EntityId
PromoteElement::ChildNodeRequest::suggested_node_id(int childNumber) const
{
  return idProcPairsFromAllProcs_[childNumber][0].second;
}
//--------------------------------------------------------------------------
std::vector<size_t>
PromoteElement::ChildNodeRequest::determine_child_node_ordinal(
  const stk::mesh::BulkData& mesh,
  const ElementDescription& elemDesc,
  unsigned elemNumber) const
{
  const auto& elem = sharedElems_[elemNumber];
  stk::mesh::Entity const* node_rels = mesh.begin_nodes(elem);
  const size_t numNodes = mesh.num_nodes(elem);

  std::array<unsigned, 8> parent_node_ordinals;
  unsigned numParents = parentIds_.size();
  ThrowAssert(numParents <= 8);

  for (unsigned i = 0; i < numParents; ++i) {
    for (unsigned j = 0; j < numNodes; ++j) {
      if (mesh.identifier(node_rels[j]) == unsortedParentIds_[i]) {
        parent_node_ordinals[i] = j;
      }
    }
  }
  unsortedParentOrdinals_[elemNumber].resize(numParents);
  std::copy(parent_node_ordinals.begin(),
    parent_node_ordinals.begin()+numParents,
    unsortedParentOrdinals_[elemNumber].begin()
  );

  std::vector<size_t> reorderedOrdinals;
  for (const auto& relation : elemDesc.addedConnectivities) {
    if (relation.second.size() == numParents) {
      std::vector<size_t> parentOrdinals(numParents);
      std::copy(parent_node_ordinals.begin(),
        parent_node_ordinals.begin()+numParents,
        parentOrdinals.begin()
      );

      if (parentOrdinals == relation.second) {
        reorderedOrdinals = relation.first;
        childOrdinalsForElem_[elemNumber] = relation.first;
        break;
      }
      else {
        bool isPermutation = std::is_permutation(
          relation.second.begin(),
          relation.second.end(),
          parent_node_ordinals.begin()
        );

        if (isPermutation) {
          reorderedOrdinals = relation.first;
          childOrdinalsForElem_[elemNumber] = relation.first;
          break;
        }
      }
    }
  }
  return reorderedOrdinals;
}


} // namespace naluUnit
}  // namespace sierra

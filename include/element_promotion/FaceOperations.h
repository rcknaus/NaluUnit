#ifndef FaceOperations_h
#define FaceOperations_h

#include <stddef.h>
#include <vector>

namespace sierra {
namespace naluUnit {

  template<class T>
  constexpr T ipow(const T base, unsigned const exponent)
  {
    return (exponent == 0) ? 1 : (base * ipow(base, exponent-1));
  }

  template<typename T> bool
  parents_are_reversed(
    const std::vector<T>& test,
    const std::vector<T>& gold)
  {
    const unsigned numParents = gold.size();
    if (test.size() != numParents) {
      return false;
    }

    bool parentAreFlipped = true;
    for (unsigned j = 0; j < numParents; ++j) {
      if (gold.at(j) != test.at(numParents-1-j)) {
        return false;
      }
    }
    return parentAreFlipped;
  }

  template<typename T> void
  flip_x(
    std::vector<T>& childOrdinals,
    unsigned size1D)
  {
    auto copy = childOrdinals;
    for (unsigned j = 0; j < size1D; ++j) {
      for (unsigned i = 0; i < size1D; ++i) {
        int ir = size1D-i-1;
        childOrdinals.at(i+(size1D)*j) = copy.at(ir+(size1D)*j);
      }
    }
  }

  template<typename T> bool
  parents_are_flipped_x(
    const std::vector<T>& test,
    const std::vector<T>& gold,
    unsigned size1D)
  {
    if (test.size() != gold.size() || test.size() != size1D*size1D) {
      return false;
    }

    bool parentAreFlipped = true;
    for (unsigned j = 0; j < size1D; ++j) {
      for (unsigned i = 0; i < size1D; ++i) {
        int ir = size1D-i-1;
        if (gold.at(i+(size1D)*j) != test.at(ir+(size1D)*j)) {
          return false;
        }
      }
    }
    return parentAreFlipped;
  }

  template<typename T> void
  flip_y(
    std::vector<T>& childOrdinals,
    unsigned size1D)
  {
    auto copy = childOrdinals;
    for (unsigned j = 0; j < size1D; ++j) {
      for (unsigned i = 0; i < size1D; ++i) {
        int jr = size1D-j-1;
        childOrdinals.at(i+(size1D)*j) = copy.at(i+(size1D)*jr);
      }
    }
  }

  template<typename T> bool
  parents_are_flipped_y(
    const std::vector<T>& test,
    const std::vector<T>& gold,
    unsigned size1D)
  {
    if (test.size() != gold.size() || test.size() != size1D*size1D) {
      return false;
    }
    bool parentAreFlipped = true;
    for (unsigned j = 0; j < size1D; ++j) {
      for (unsigned i = 0; i < size1D; ++i) {
        int jr = size1D-j-1;
        if (gold.at(i+(size1D)*j) != test.at(i+(size1D)*jr)) {
          return false;
        }
      }
    }
    return parentAreFlipped;
  }

  template<typename T> void
  transpose_ordinals(
    std::vector<T>& childOrdinals,
    unsigned size1D)
  {
    auto copy = childOrdinals;
    for (unsigned j = 0; j < size1D; ++j) {
      for (unsigned i = 0; i < size1D; ++i) {
        childOrdinals.at(i+(size1D)*j) = copy.at(j+(size1D)*i);
      }
    }
  }

} // namespace naluUnit
} // namespace Sierra

#endif

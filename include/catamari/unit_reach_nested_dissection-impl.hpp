/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_UNIT_REACH_NESTED_DISSECTION_IMPL_H_
#define CATAMARI_UNIT_REACH_NESTED_DISSECTION_IMPL_H_

#include "catamari/unit_reach_nested_dissection.hpp"

namespace catamari {

template <typename T>
struct Extent2D {
  T x_beg;
  T x_end;
  T y_beg;
  T y_end;
};

template <typename T>
struct Extent3D {
  T x_beg;
  T x_end;
  T y_beg;
  T y_end;
  T z_beg;
  T z_end;
};

void UnitReachNestedDissection2DRecursion(Int num_x_elements,
                                          Int num_y_elements, Int offset,
                                          const Extent2D<Int>& box,
                                          Int* supernode_index,
                                          SymmetricOrdering* ordering) {
  const Int y_stride = num_x_elements + 1;
  const Int min_cut_size = 5;

  // Determine which dimension to cut (if any).
  const Int x_size = box.x_end - box.x_beg;
  const Int y_size = box.y_end - box.y_beg;
  const Int max_size = std::max(x_size, y_size);
  CATAMARI_ASSERT(max_size > 0, "The maximum size was non-positive.");
  if (max_size < min_cut_size) {
    // Do not split.
    for (Int y = box.y_beg; y < box.y_end; ++y) {
      for (Int x = box.x_beg; x < box.x_end; ++x) {
        ordering->inverse_permutation[offset++] = x + y * y_stride;
      }
    }

    const Int supernode_size =
        (box.y_end - box.y_beg) * (box.x_end - box.x_beg);
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);

    return;
  }

  if (x_size == max_size) {
    // Cut the x dimension.
    const Int x_cut = box.x_beg + (box.x_end - box.x_beg) / 2;
    const Extent2D<Int> left_box{box.x_beg, x_cut, box.y_beg, box.y_end};
    const Extent2D<Int> right_box{x_cut + 1, box.x_end, box.y_beg, box.y_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset +
        (left_box.x_end - left_box.x_beg) * (left_box.y_end - left_box.y_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg);

    // Fill the left child.
    UnitReachNestedDissection2DRecursion(num_x_elements, num_y_elements,
                                         left_offset, left_box, supernode_index,
                                         ordering);
    const Int left_child = *supernode_index - 1;

    // Fill the right child.
    UnitReachNestedDissection2DRecursion(num_x_elements, num_y_elements,
                                         right_offset, right_box,
                                         supernode_index, ordering);
    const Int right_child = *supernode_index - 1;

    // Fill the separator.
    offset = cut_offset;
    for (Int y = box.y_beg; y < box.y_end; ++y) {
      ordering->inverse_permutation[offset++] = x_cut + y * y_stride;
    }

    const Int supernode = *supernode_index;
    const Int supernode_size = box.y_end - box.y_beg;
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents[left_child] = supernode;
    ordering->permuted_assembly_forest.parents[right_child] = supernode;
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);
  } else {
    // Cut the y dimension.
    const Int y_cut = box.y_beg + (box.y_end - box.y_beg) / 2;
    const Extent2D<Int> left_box{box.x_beg, box.x_end, box.y_beg, y_cut};
    const Extent2D<Int> right_box{box.x_beg, box.x_end, y_cut + 1, box.y_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset +
        (left_box.x_end - left_box.x_beg) * (left_box.y_end - left_box.y_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg);

    // Fill the left child.
    UnitReachNestedDissection2DRecursion(num_x_elements, num_y_elements,
                                         left_offset, left_box, supernode_index,
                                         ordering);
    const Int left_child = *supernode_index - 1;

    // Fill the right child.
    UnitReachNestedDissection2DRecursion(num_x_elements, num_y_elements,
                                         right_offset, right_box,
                                         supernode_index, ordering);
    const Int right_child = *supernode_index - 1;

    // Fill the separator.
    offset = cut_offset;
    for (Int x = box.x_beg; x < box.x_end; ++x) {
      ordering->inverse_permutation[offset++] = x + y_cut * y_stride;
    }

    const Int supernode = *supernode_index;
    const Int supernode_size = box.x_end - box.x_beg;
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents[left_child] = supernode;
    ordering->permuted_assembly_forest.parents[right_child] = supernode;
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);
  }
}

void UnitReachNestedDissection2D(Int num_x_elements, Int num_y_elements,
                                 SymmetricOrdering* ordering) {
  const Int num_rows = (num_x_elements + 1) * (num_y_elements + 1);
  ordering->permutation.resize(num_rows);
  ordering->inverse_permutation.resize(num_rows);
  ordering->permuted_supernode_sizes.reserve(num_rows);
  ordering->permuted_assembly_forest.parents.reserve(num_rows);

  Int offset = 0;
  Extent2D<Int> box{0, num_x_elements + 1, 0, num_y_elements + 1};
  Int supernode_index = 0;
  UnitReachNestedDissection2DRecursion(num_x_elements, num_y_elements, offset,
                                       box, &supernode_index, ordering);

  // Invert the inverse permutation.
  for (Int row = 0; row < num_rows; ++row) {
    ordering->permutation[ordering->inverse_permutation[row]] = row;
  }

  quotient::ChildrenFromParents(
      ordering->permuted_assembly_forest.parents,
      &ordering->permuted_assembly_forest.children,
      &ordering->permuted_assembly_forest.child_offsets);
}

void UnitReachNestedDissection3DRecursion(Int num_x_elements,
                                          Int num_y_elements,
                                          Int num_z_elements, Int offset,
                                          const Extent3D<Int>& box,
                                          Int* supernode_index,
                                          SymmetricOrdering* ordering) {
  const Int y_stride = num_x_elements + 1;
  const Int z_stride = y_stride * (num_y_elements + 1);
  const Int min_cut_size = 5;

  // Determine which dimension to cut (if any).
  const Int x_size = box.x_end - box.x_beg;
  const Int y_size = box.y_end - box.y_beg;
  const Int z_size = box.z_end - box.z_beg;
  const Int max_size = std::max(std::max(x_size, y_size), z_size);
  CATAMARI_ASSERT(max_size > 0, "The maximum size was non-positive.");
  if (max_size < min_cut_size) {
    // Do not split.
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int y = box.y_beg; y < box.y_end; ++y) {
        for (Int x = box.x_beg; x < box.x_end; ++x) {
          ordering->inverse_permutation[offset++] =
              x + y * y_stride + z * z_stride;
        }
      }
    }

    const Int supernode_size = (box.z_end - box.z_beg) *
                               (box.y_end - box.y_beg) *
                               (box.x_end - box.x_beg);
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);

    return;
  }

  if (x_size == max_size) {
    // Cut the x dimension.
    const Int x_cut = box.x_beg + (box.x_end - box.x_beg) / 2;
    const Extent3D<Int> left_box{box.x_beg, x_cut,     box.y_beg,
                                 box.y_end, box.z_beg, box.z_end};
    const Extent3D<Int> right_box{x_cut + 1, box.x_end, box.y_beg,
                                  box.y_end, box.z_beg, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, left_offset, left_box,
                                         supernode_index, ordering);
    const Int left_child = *supernode_index - 1;

    // Fill the right child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, right_offset,
                                         right_box, supernode_index, ordering);
    const Int right_child = *supernode_index - 1;

    // Fill the separator.
    offset = cut_offset;
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int y = box.y_beg; y < box.y_end; ++y) {
        ordering->inverse_permutation[offset++] =
            x_cut + y * y_stride + z * z_stride;
      }
    }

    const Int supernode = *supernode_index;
    const Int supernode_size =
        (box.z_end - box.z_beg) * (box.y_end - box.y_beg);
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents[left_child] = supernode;
    ordering->permuted_assembly_forest.parents[right_child] = supernode;
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);
  } else if (y_size == max_size) {
    // Cut the y dimension.
    const Int y_cut = box.y_beg + (box.y_end - box.y_beg) / 2;
    const Extent3D<Int> left_box{box.x_beg, box.x_end, box.y_beg,
                                 y_cut,     box.z_beg, box.z_end};
    const Extent3D<Int> right_box{box.x_beg, box.x_end, y_cut + 1,
                                  box.y_end, box.z_beg, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, left_offset, left_box,
                                         supernode_index, ordering);
    const Int left_child = *supernode_index - 1;

    // Fill the right child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, right_offset,
                                         right_box, supernode_index, ordering);
    const Int right_child = *supernode_index - 1;

    // Fill the separator.
    offset = cut_offset;
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int x = box.x_beg; x < box.x_end; ++x) {
        ordering->inverse_permutation[offset++] =
            x + y_cut * y_stride + z * z_stride;
      }
    }

    const Int supernode = *supernode_index;
    const Int supernode_size =
        (box.z_end - box.z_beg) * (box.x_end - box.x_beg);
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents[left_child] = supernode;
    ordering->permuted_assembly_forest.parents[right_child] = supernode;
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);
  } else {
    // Cut the z dimension.
    const Int z_cut = box.z_beg + (box.z_end - box.z_beg) / 2;
    const Extent3D<Int> left_box{box.x_beg, box.x_end, box.y_beg,
                                 box.y_end, box.z_beg, z_cut};
    const Extent3D<Int> right_box{box.x_beg, box.x_end, box.y_beg,
                                  box.y_end, z_cut + 1, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, left_offset, left_box,
                                         supernode_index, ordering);
    const Int left_child = *supernode_index - 1;

    // Fill the right child.
    UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                         num_z_elements, right_offset,
                                         right_box, supernode_index, ordering);
    const Int right_child = *supernode_index - 1;

    // Fill the separator.
    offset = cut_offset;
    for (Int y = box.y_beg; y < box.y_end; ++y) {
      for (Int x = box.x_beg; x < box.x_end; ++x) {
        ordering->inverse_permutation[offset++] =
            x + y * y_stride + z_cut * z_stride;
      }
    }

    const Int supernode = *supernode_index;
    const Int supernode_size =
        (box.y_end - box.y_beg) * (box.x_end - box.x_beg);
    ordering->permuted_supernode_sizes.push_back(supernode_size);
    ordering->permuted_assembly_forest.parents[left_child] = supernode;
    ordering->permuted_assembly_forest.parents[right_child] = supernode;
    ordering->permuted_assembly_forest.parents.push_back(-1);
    ++(*supernode_index);
  }
}

void UnitReachNestedDissection3D(Int num_x_elements, Int num_y_elements,
                                 Int num_z_elements,
                                 SymmetricOrdering* ordering) {
  const Int num_rows =
      (num_x_elements + 1) * (num_y_elements + 1) * (num_z_elements + 1);
  ordering->permutation.resize(num_rows);
  ordering->inverse_permutation.resize(num_rows);
  ordering->permuted_supernode_sizes.reserve(num_rows);
  ordering->permuted_assembly_forest.parents.reserve(num_rows);

  Int offset = 0;
  Extent3D<Int> box{0, num_x_elements + 1, 0, num_y_elements + 1,
                    0, num_z_elements + 1};
  Int supernode_index = 0;
  UnitReachNestedDissection3DRecursion(num_x_elements, num_y_elements,
                                       num_z_elements, offset, box,
                                       &supernode_index, ordering);

  // Invert the inverse permutation.
  for (Int row = 0; row < num_rows; ++row) {
    ordering->permutation[ordering->inverse_permutation[row]] = row;
  }

  quotient::ChildrenFromParents(
      ordering->permuted_assembly_forest.parents,
      &ordering->permuted_assembly_forest.children,
      &ordering->permuted_assembly_forest.child_offsets);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_UNIT_REACH_NESTED_DISSECTION_IMPL_H_

/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_UNIT_REACH_NESTED_DISSECTION_H_
#define CATAMARI_UNIT_REACH_NESTED_DISSECTION_H_

#include "catamari/symmetric_ordering.hpp"

namespace catamari {

// Analytically performs nested dissection on a
// 'num_x_elements x num_y_elements' grid where each vertex only touches its
// nearest neighbors.
void UnitReach2DNestedDissection(Int num_x_elements, Int num_y_elements,
                                 SymmetricOrdering* ordering);

// Analytically performs nested dissection on a
// 'num_x_elements x num_y_elements x num_z_elements' grid where each vertex
// only touches its nearest neighbors.
void UnitReach3DNestedDissection(Int num_x_elements, Int num_y_elements,
                                 Int num_z_elements,
                                 SymmetricOrdering* ordering);

}  // namespace catamari

#include "catamari/unit_reach_nested_dissection-impl.hpp"

#endif  // ifndef CATAMARI_UNIT_REACH_NESTED_DISSECTION_H_

/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_ENABLE_IF_H_
#define CATAMARI_ENABLE_IF_H_

#include <type_traits>

namespace catamari {

// For overloading function definitions using type traits. For example:
//
//   template<typename T, typename=EnableIf<std::is_integral<T>>>
//   int Log(T value);
//
//   template<typename T, typename=DisableIf<std::is_integral<T>>>
//   double Log(T value);
//
// would lead to the 'Log' function returning an 'int' for any integral type
// and a 'double' for any non-integral type.
template<typename Condition, class T = void>
using EnableIf = typename std::enable_if<Condition::value, T>::type;
template<typename Condition, class T = void>
using DisableIf = typename std::enable_if<!Condition::value, T>::type;

} // namespace catamari

#endif // ifndef CATAMARI_ENABLE_IF_H_

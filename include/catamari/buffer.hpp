/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BUFFER_H_
#define CATAMARI_BUFFER_H_

#include <memory>

#include "quotient/integers.hpp"

namespace catamari {

// A simple stand-in for some of the functionality of std::vector that avoids
// explicit initialization in the single-argument versions of the constructor
// and in resize. In exchange, it does not have 'push_back' functionality.
template <typename T>
class Buffer {
 public:
  // Constructs a zero-length buffer.
  Buffer();

  // Constructs a buffer of the given length without initializing the data.
  Buffer(Int num_elements);

  // Constructs a buffer of the given length where each entry is initialized
  // to the specified value.
  Buffer(Int num_elements, const T& value);

  // Resizes the buffer to store the given number of elements and avoids
  // initializing values where possible.
  void Resize(Int num_elements);

  // Resizes the buffer to store the given number of elements and initializes
  // *all* entries in the new buffer to the specified value.
  void Resize(Int num_elements, const T& value);

  // Returns the current length of the buffer.
  Int Size() const;

  // Returns the current capacity of the buffer.
  Int Capacity() const;

  // Returns a mutable pointer to the underlying buffer of entries.
  T* Data();

  // Returns an immutable pointer to the underlying buffer of entries.
  const T* Data() const;

  // Returns a mutable reference to the given entry of the buffer.
  T& operator[](Int index);

  // Returns an immutable reference to the given entry of the buffer.
  const T& operator[](Int index) const;

 private:
  // The number of entries stored by the buffer.
  Int size_;

  // The number of entries that could have been stored by 'data_'.
  Int capacity_;

  // The underlying pointer for storing entries in the buffer.
  std::unique_ptr<T> data_;
};

}  // namespace catamari

#include "catamari/buffer-impl.hpp"

#endif  // ifndef CATAMARI_BUFFER_H_

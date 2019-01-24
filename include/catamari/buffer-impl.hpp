/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BUFFER_IMPL_H_
#define CATAMARI_BUFFER_IMPL_H_

#include "catamari/buffer.hpp"

namespace catamari {

template <typename T>
Buffer<T>::Buffer() : size_(0), capacity_(0) {}

template <typename T>
Buffer<T>::Buffer(Int num_elements)
    : size_(num_elements),
      capacity_(num_elements),
      data_(new T[num_elements]) {}

template <typename T>
Buffer<T>::Buffer(Int num_elements, const T& value)
    : size_(num_elements), capacity_(num_elements), data_(new T[num_elements]) {
  // TODO(Jack Poulson): Decide if a variant of 'new' would be faster.
  T* data = data_.get();
  for (Int index = 0; index < num_elements; ++index) {
    data[index] = value;
  }
}

template <typename T>
void Buffer<T>::Resize(Int num_elements) {
  if (num_elements > capacity_) {
    data_.reset(new T[num_elements]);
    capacity_ = num_elements;
  }
  size_ = num_elements;
}

template <typename T>
void Buffer<T>::Resize(Int num_elements, const T& value) {
  if (num_elements > capacity_) {
    data_.reset(new T[num_elements]);
    capacity_ = num_elements;
  }
  size_ = num_elements;

  T* data = data_.get();
  for (Int index = 0; index < num_elements; ++index) {
    data[index] = value;
  }
}

template <typename T>
Int Buffer<T>::Size() const {
  return size_;
}

template <typename T>
Int Buffer<T>::Capacity() const {
  return capacity_;
}

template <typename T>
T* Buffer<T>::Data() {
  return data_.get();
}

template <typename T>
const T* Buffer<T>::Data() const {
  return data_.get();
}

template <typename T>
T& Buffer<T>::operator[](Int index) {
  T* data = data_.get();
  return data[index];
}

template <typename T>
const T& Buffer<T>::operator[](Int index) const {
  const T* data = data_.get();
  return data[index];
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BUFFER_IMPL_H_

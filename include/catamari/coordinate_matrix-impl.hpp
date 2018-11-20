/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COORDINATE_MATRIX_IMPL_H_
#define CATAMARI_COORDINATE_MATRIX_IMPL_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "catamari/complex.hpp"
#include "catamari/integers.hpp"
#include "catamari/macros.hpp"
#include "catamari/matrix_market.hpp"

#include "catamari/coordinate_matrix.hpp"

namespace catamari {

template <class Field>
CoordinateMatrix<Field>::CoordinateMatrix() : num_rows_(0), num_columns_(0) {}

template <class Field>
CoordinateMatrix<Field>::CoordinateMatrix(
    const CoordinateMatrix<Field>& matrix) {
  if (&matrix == this) {
    return;
  }
  *this = matrix;
}

template <class Field>
const CoordinateMatrix<Field>& CoordinateMatrix<Field>::operator=(
    const CoordinateMatrix<Field>& matrix) {
  if (&matrix == this) {
    return *this;
  }

  num_rows_ = matrix.num_rows_;
  num_columns_ = matrix.num_columns_;
  entries_ = matrix.entries_;
  row_entry_offsets_ = matrix.row_entry_offsets_;
  entries_to_add_ = matrix.entries_to_add_;
  entries_to_remove_ = matrix.entries_to_remove_;

  return *this;
}

template <class Field>
std::unique_ptr<CoordinateMatrix<Field>>
CoordinateMatrix<Field>::FromMatrixMarket(const std::string& filename,
                                          bool skip_explicit_zeros,
                                          EntryMask mask) {
  std::unique_ptr<CoordinateMatrix<Field>> result;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open " << filename << std::endl;
    return result;
  }

  // Fill the description of the Matrix Market data.
  MatrixMarketDescription description;
  if (!quotient::ReadMatrixMarketDescription(file, &description)) {
    return result;
  }

  result.reset(new CoordinateMatrix<Field>);
  if (description.format == quotient::kMatrixMarketFormatArray) {
    // Read the size of the matrix.
    Int num_rows, num_columns;
    if (!quotient::ReadMatrixMarketArrayMetadata(description, file, &num_rows,
                                                 &num_columns)) {
      result.reset();
      return result;
    }

    // Fill a fully-connected graph.
    result->Resize(num_rows, num_columns);
    result->ReserveEntryAdditions(num_rows * num_columns);
    for (Int column = 0; column < num_columns; ++column) {
      for (Int row = 0; row < num_rows; ++row) {
        Field value;
        if (!ReadMatrixMarketArrayValue(description, file, &value)) {
          result.reset();
          return result;
        }
        result->QueueEntryAddition(row, column, value);
      }
    }
    result->FlushEntryQueues();
    return result;
  }

  // Read in the number of matrix dimensions and the number of entries specified
  // in the file.
  Int num_rows, num_columns, num_entries;
  if (!quotient::ReadMatrixMarketCoordinateMetadata(
          description, file, &num_rows, &num_columns, &num_entries)) {
    result.reset();
    return result;
  }

  // Fill in the entries.
  Int num_skipped_entries = 0;
  const Int num_entries_bound =
      description.symmetry == quotient::kMatrixMarketSymmetryGeneral
          ? num_entries
          : 2 * num_entries;
  result->Resize(num_rows, num_columns);
  result->ReserveEntryAdditions(num_entries_bound);
  for (Int entry_index = 0; entry_index < num_entries; ++entry_index) {
    Int row, column;
    Field value;
    if (!ReadMatrixMarketCoordinateEntry(description, file, &row, &column,
                                         &value)) {
      result.reset();
      return result;
    }

    if ((mask == quotient::kEntryMaskLowerTriangle && row < column) ||
        (mask == quotient::kEntryMaskUpperTriangle && row > column)) {
      continue;
    }

    if (skip_explicit_zeros) {
      // Skip this entry if it is numerically zero.
      if (value == Field(0)) {
        ++num_skipped_entries;
        continue;
      }
    }

    result->QueueEntryAddition(row, column, value);
    if (row != column) {
      if (description.symmetry == quotient::kMatrixMarketSymmetrySymmetric) {
        result->QueueEntryAddition(column, row, value);
      } else if (description.symmetry ==
                 quotient::kMatrixMarketSymmetryHermitian) {
        result->QueueEntryAddition(column, row, Conjugate(value));
      } else if (description.symmetry ==
                 quotient::kMatrixMarketSymmetrySkewSymmetric) {
        result->QueueEntryAddition(column, row, -value);
      }
    }
  }
  result->FlushEntryQueues();

  if (num_skipped_entries) {
    std::cout << "Skipped " << num_skipped_entries << " explicitly zeros."
              << std::endl;
  }

  return result;
}

template <class Field>
void CoordinateMatrix<Field>::ToMatrixMarket(
    const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open " << filename << std::endl;
    return;
  }

  // Write the header.
  {
    const std::string field_string =
        IsComplex<Field>::value ? quotient::kMatrixMarketFieldComplexString
                                : quotient::kMatrixMarketFieldRealString;
    std::ostringstream os;
    os << quotient::kMatrixMarketStampString << " "
       << quotient::kMatrixMarketObjectMatrixString << " "
       << quotient::kMatrixMarketFormatCoordinateString << " " << field_string
       << " " << quotient::kMatrixMarketSymmetryGeneralString << "\n";
    file << os.str();
  }

  // Write the size information.
  {
    std::ostringstream os;
    os << NumRows() << " " << NumColumns() << " " << NumEntries() << "\n";
    file << os.str();
  }

  // Write out the entries.
  const std::vector<MatrixEntry<Field>>& entries = Entries();
  for (const MatrixEntry<Field>& entry : entries) {
    // We must convert from 0-based to 1-based indexing.
    std::ostringstream os;
    os << entry.row + 1 << " " << entry.column + 1 << " ";
    if (IsComplex<Field>::value) {
      os << RealPart(entry.value) << " " << ImagPart(entry.value) << "\n";
    } else {
      os << RealPart(entry.value) << "\n";
    }
    file << os.str();
  }
}

template <class Field>
CoordinateMatrix<Field>::~CoordinateMatrix() {}

template <class Field>
Int CoordinateMatrix<Field>::NumRows() const CATAMARI_NOEXCEPT {
  return num_rows_;
}

template <class Field>
Int CoordinateMatrix<Field>::NumColumns() const CATAMARI_NOEXCEPT {
  return num_columns_;
}

template <class Field>
Int CoordinateMatrix<Field>::NumEntries() const CATAMARI_NOEXCEPT {
  return entries_.size();
}

template <class Field>
void CoordinateMatrix<Field>::Empty(bool free_resources) {
  if (free_resources) {
    SwapClearVector(&entries_);
    SwapClearVector(&row_entry_offsets_);
    SwapClearVector(&entries_to_add_);
    SwapClearVector(&entries_to_remove_);
  } else {
    entries_.clear();
    entries_to_add_.clear();
    entries_to_remove_.clear();
  }

  num_rows_ = num_columns_ = 0;

  // Create a trivial row offset vector.
  row_entry_offsets_.resize(1);
  row_entry_offsets_[0] = 0;
}

template <class Field>
void CoordinateMatrix<Field>::Resize(Int num_rows, Int num_columns) {
  if (num_rows == num_rows_ && num_columns == num_columns_) {
    return;
  }

  num_rows_ = num_rows;
  num_columns_ = num_columns;

  entries_.clear();
  entries_to_add_.clear();
  entries_to_remove_.clear();

  row_entry_offsets_.resize(num_rows + 1);
  for (Int row = 0; row <= num_rows; ++row) {
    row_entry_offsets_[row] = 0;
  }
}

template <class Field>
void CoordinateMatrix<Field>::ReserveEntryAdditions(Int max_entry_additions) {
  entries_to_add_.reserve(max_entry_additions);
}

template <class Field>
void CoordinateMatrix<Field>::QueueEntryAddition(Int row, Int column,
                                                 const Field& value) {
  CATAMARI_ASSERT(entries_to_add_.size() != entries_to_add_.capacity(),
                  "WARNING: Pushing back without first reserving space.");
  CATAMARI_ASSERT(row >= 0 && row < num_rows_,
                  "ERROR: Row index was out of bounds.");
  CATAMARI_ASSERT(column >= 0 && column < num_columns_,
                  "ERROR: Column index was out of bounds.");
  entries_to_add_.emplace_back(row, column, value);
}

template <class Field>
void CoordinateMatrix<Field>::FlushEntryAdditionQueue(
    bool update_row_entry_offsets) {
  if (!entries_to_add_.empty()) {
    // Sort and combine the list of entries to add.
    std::sort(entries_to_add_.begin(), entries_to_add_.end());
    CombineSortedEntries(&entries_to_add_);

    // Perform a merge sort and then combine entries with the same indices.
    const std::vector<MatrixEntry<Field>> entries_copy(entries_);
    entries_.resize(0);
    entries_.resize(entries_copy.size() + entries_to_add_.size());
    std::merge(entries_copy.begin(), entries_copy.end(),
               entries_to_add_.begin(), entries_to_add_.end(),
               entries_.begin());
    SwapClearVector(&entries_to_add_);
    CombineSortedEntries(&entries_);
  }

  if (update_row_entry_offsets) {
    UpdateRowEntryOffsets();
  }
}

template <class Field>
void CoordinateMatrix<Field>::ReserveEntryRemovals(Int max_entry_removals) {
  entries_to_remove_.reserve(max_entry_removals);
}

template <class Field>
void CoordinateMatrix<Field>::QueueEntryRemoval(Int row, Int column) {
  CATAMARI_ASSERT(row >= 0 && row < num_rows_,
                  "ERROR: Row index was out of bounds.");
  CATAMARI_ASSERT(column >= 0 && column < num_columns_,
                  "ERROR: Column index was out of bounds.");
  entries_to_remove_.emplace_back(row, column);
}

template <class Field>
void CoordinateMatrix<Field>::FlushEntryRemovalQueue(
    bool update_row_entry_offsets) {
  if (!entries_to_remove_.empty()) {
    // Sort and erase duplicates from the list of edges to be removed.
    std::sort(entries_to_remove_.begin(), entries_to_remove_.end());
    quotient::EraseDuplicatesInSortedVector(&entries_to_remove_);

    const Int num_entries = entries_.size();
    Int num_packed = 0;
    for (Int index = 0; index < num_entries; ++index) {
      GraphEdge edge{entries_[index].row, entries_[index].column};
      auto iter = std::lower_bound(entries_to_remove_.begin(),
                                   entries_to_remove_.end(), edge);
      if (iter == entries_to_remove_.end() || *iter != edge) {
        // The current entry should be kept, so pack it from the left.
        entries_[num_packed++] = entries_[index];
      }
    }
    entries_.resize(num_packed);
    SwapClearVector(&entries_to_remove_);
  }

  if (update_row_entry_offsets) {
    UpdateRowEntryOffsets();
  }
}

template <class Field>
void CoordinateMatrix<Field>::FlushEntryQueues() {
  if (EntryQueuesAreEmpty()) {
    // Skip the recomputation of the row offsets.
    return;
  }
  FlushEntryRemovalQueue(false /* update_row_entry_offsets */);
  FlushEntryAdditionQueue(true /* update_row_entry_offsets */);
}

template <class Field>
bool CoordinateMatrix<Field>::EntryQueuesAreEmpty() const CATAMARI_NOEXCEPT {
  return entries_to_add_.empty() && entries_to_remove_.empty();
}

template <class Field>
void CoordinateMatrix<Field>::AddEntry(Int row, Int column,
                                       const Field& value) {
  ReserveEntryAdditions(1);
  QueueEntryAddition(row, column, value);
  FlushEntryQueues();
}

template <class Field>
void CoordinateMatrix<Field>::RemoveEntry(Int row, Int column) {
  ReserveEntryRemovals(1);
  QueueEntryRemoval(row, column);
  FlushEntryQueues();
}

template <class Field>
const MatrixEntry<Field>& CoordinateMatrix<Field>::Entry(Int entry_index) const
    CATAMARI_NOEXCEPT {
#ifdef CATAMARI_DEBUG
  return entries_.at(entry_index);
#else
  return entries_[entry_index];
#endif
}

template <class Field>
const std::vector<MatrixEntry<Field>>& CoordinateMatrix<Field>::Entries() const
    CATAMARI_NOEXCEPT {
  return entries_;
}

template <class Field>
Int CoordinateMatrix<Field>::RowEntryOffset(Int row) const CATAMARI_NOEXCEPT {
  CATAMARI_ASSERT(
      EntryQueuesAreEmpty(),
      "Tried to retrieve a row edge offset when entry queues weren't empty");
#ifdef CATAMARI_DEBUG
  return row_entry_offsets_.at(row);
#else
  return row_entry_offsets_[row];
#endif
}

template <class Field>
Int CoordinateMatrix<Field>::EntryOffset(Int row,
                                         Int column) const CATAMARI_NOEXCEPT {
  const Int row_entry_offset = RowEntryOffset(row);
  const Int next_row_entry_offset = RowEntryOffset(row + 1);
  const MatrixEntry<Field> target_entry{row, column, Field{0}};
  auto iter =
      std::lower_bound(entries_.begin() + row_entry_offset,
                       entries_.begin() + next_row_entry_offset, target_entry);
  return iter - entries_.begin();
}

template <class Field>
bool CoordinateMatrix<Field>::EntryExists(Int row,
                                          Int column) const CATAMARI_NOEXCEPT {
  const Int index = EntryOffset(row, column);
  const MatrixEntry<Field>& entry = Entry(index);
  return entry.row == row && entry.column == column;
}

template <class Field>
Int CoordinateMatrix<Field>::NumRowNonzeros(Int row) const CATAMARI_NOEXCEPT {
  return RowEntryOffset(row + 1) - RowEntryOffset(row);
}

template <class Field>
std::unique_ptr<quotient::CoordinateGraph>
CoordinateMatrix<Field>::CoordinateGraph() const CATAMARI_NOEXCEPT {
  std::unique_ptr<quotient::CoordinateGraph> graph(
      new quotient::CoordinateGraph);
  graph->AsymmetricResize(NumRows(), NumColumns());
  graph->ReserveEdgeAdditions(NumEntries());
  for (const MatrixEntry<Field>& entry : Entries()) {
    graph->QueueEdgeAddition(entry.row, entry.column);
  }
  graph->FlushEdgeQueues();
  return graph;
}

template <class Field>
void CoordinateMatrix<Field>::UpdateRowEntryOffsets() {
  const Int num_entries = entries_.size();
  row_entry_offsets_.resize(num_rows_ + 1);
  Int row_entry_offset = 0;
  Int prev_row = -1;
  for (Int entry_index = 0; entry_index < num_entries; ++entry_index) {
    const Int row = entries_[entry_index].row;
    CATAMARI_ASSERT(row >= prev_row, "Rows were not properly sorted.");

    // Fill in the row offsets from prev_row to row - 1.
    for (; prev_row < row; ++prev_row) {
      row_entry_offsets_[row_entry_offset++] = entry_index;
    }
  }

  // Fill in the end of the row offset buffer.
  for (; row_entry_offset <= num_rows_; ++row_entry_offset) {
    row_entry_offsets_[row_entry_offset] = num_entries;
  }
}

template <typename Field>
void CoordinateMatrix<Field>::CombineSortedEntries(
    std::vector<MatrixEntry<Field>>* entries) {
  Int last_row = -1, last_column = -1;
  Int num_packed = 0;
  for (std::size_t index = 0; index < entries->size(); ++index) {
    const MatrixEntry<Field>& entry = (*entries)[index];

    if (entry.row == last_row && entry.column == last_column) {
      (*entries)[num_packed - 1].value += entry.value;
    } else {
      last_row = entry.row;
      last_column = entry.column;
      (*entries)[num_packed++] = entry;
    }
  }
  entries->resize(num_packed);
}

template <class Field>
void PrintCoordinateMatrix(const CoordinateMatrix<Field>& matrix,
                           const std::string& label) {
  std::cout << label << ":\n";
  for (const MatrixEntry<Field>& entry : matrix.Entries()) {
    std::cout << entry.row << " " << entry.column << " " << entry.value << "\n";
  }
  std::cout << std::endl;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_COORDINATE_MATRIX_IMPL_H_

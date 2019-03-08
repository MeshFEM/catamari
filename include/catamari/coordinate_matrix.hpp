/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COORDINATE_MATRIX_H_
#define CATAMARI_COORDINATE_MATRIX_H_

#include "catamari/buffer.hpp"
#include "catamari/integers.hpp"
#include "catamari/macros.hpp"
#include "quotient/coordinate_graph.hpp"

namespace catamari {

using quotient::EntryMask;
using quotient::GraphEdge;
using quotient::SwapClearVector;

// A tuple of the row, column, and value of a nonzero in a sparse matrix.
template <class Field>
struct MatrixEntry {
  // The row index of the entry.
  Int row;

  // The column index of the entry.
  Int column;

  // The numerical value of the entry.
  Field value;

  // A trivial constructor (required for STL sorting).
  MatrixEntry() = default;

  // A standard constructor (required for emplacement).
  MatrixEntry(Int row_, Int column_, const Field& value_)
      : row(row_), column(column_), value(value_) {}

  // A copy constructor.
  MatrixEntry(const MatrixEntry<Field>& other) = default;

  // A partial ordering that ignores the floating-point value in comparisons.
  bool operator<(const MatrixEntry<Field>& other) const {
    return row < other.row || (row == other.row && column < other.column);
  }

  // An equality test that requires all members being equal.
  bool operator==(const MatrixEntry<Field>& other) const {
    return row == other.row && column == other.column && value == other.value;
  }

  // An inequality test that requires all members being equal.
  bool operator!=(const MatrixEntry<Field>& other) const {
    return !operator==(other);
  }
};

// A coordinate-format sparse matrix data structure. The primary storage is a
// lexicographically sorted Buffer<MatrixEntry<Field>> and an associated
// Buffer<Int> of row offsets (which serve the same role as in a Compressed
// Sparse Row (CSR) format). Thus, this storage scheme is a superset of the CSR
// format that explicitly stores both row and column indices for each entry.
//
// The class is designed so that the sorting and offset computation overhead
// can be amortized over batches of entry additions and removals.
//
// For example, the code block:
//
//   catamari::CoordinateMatrix<double> matrix;
//   matrix.Resize(5, 5);
//   matrix.ReserveEntryAdditions(6);
//   matrix.QueueEntryAddition(3, 4, 1.);
//   matrix.QueueEntryAddition(2, 3, 2.);
//   matrix.QueueEntryAddition(2, 0, -1.);
//   matrix.QueueEntryAddition(4, 2, -2.);
//   matrix.QueueEntryAddition(4, 4, 3.);
//   matrix.QueueEntryAddition(3, 2, 4.);
//   matrix.FlushEntryQueues();
//   const catamari::Buffer<catamari::MatrixEntry<double>>& entries =
//       matrix.Entries();
//
// would return a reference to the underlying
// catamari::Buffer<catamari::MatrixEntry<double>> of 'matrix', which should
// contain the entry sequence:
//   (2, 0, -1.), (2, 3, 2.), (3, 2, 4.), (3, 4, 1.), (4, 2, -2.), (4, 4, 3.).
//
// Similarly, subsequently running the code block:
//
//   matrix.ReserveEntryRemovals(2);
//   matrix.QueueEntryRemoval(2, 3);
//   matrix.QueueEntryRemoval(0, 4);
//   matrix.FlushEntryQueues();
//
// would modify the Buffer underlying the 'edges' reference to now
// contain the entry sequence:
//   (2, 0, -1.), (3, 2, 4.), (3, 4, 1.), (4, 2, -2.), (4, 4, 3.).
//
// TODO(Jack Poulson): Add support for 'END' index marker so that ranges
// can be easily incorporated.
template <class Field>
class CoordinateMatrix {
 public:
  // The trivial constructor.
  CoordinateMatrix();

  // The copy constructor.
  CoordinateMatrix(const CoordinateMatrix<Field>& matrix);

  // The assignment operator.
  CoordinateMatrix<Field>& operator=(const CoordinateMatrix<Field>& matrix);

  // Builds and returns a CoordinateMatrix from a Matrix Market description.
  static std::unique_ptr<CoordinateMatrix<Field>> FromMatrixMarket(
      const std::string& filename, bool skip_explicit_zeros,
      EntryMask mask = EntryMask::kEntryMaskFull);

  // Writes a copy of the CoordinateMatrix to a Matrix Market file.
  void ToMatrixMarket(const std::string& filename) const;

  // A trivial destructor.
  ~CoordinateMatrix();

  // Returns the number of rows in the matrix.
  Int NumRows() const CATAMARI_NOEXCEPT;

  // Returns the number of columns in the matrix.
  Int NumColumns() const CATAMARI_NOEXCEPT;

  // Returns the number of (usually nonzero) entries in the matrix.
  Int NumEntries() const CATAMARI_NOEXCEPT;

  // Removes all entries and changes the number of rows and columns to zero.
  void Empty();

  // Changes both the number of rows and columns.
  void Resize(Int num_rows, Int num_columns);

  // Allocates space so that up to 'max_entry_additions' calls to
  // 'QueueEntryAddition' can be performed without another memory allocation.
  void ReserveEntryAdditions(Int max_entry_additions);

  // Appends the entry to the list without putting the entry list in
  // lexicographic order or updating the row offsets.
  void QueueEntryAddition(Int row, Int column, const Field& value);

  // Allocates space so that up to 'max_entry_removals' calls to
  // 'QueueEntryRemoval' can be performed without another memory allocation.
  void ReserveEntryRemovals(Int max_entry_removals);

  // Appends the location (row, column) to the list of entries to remove.
  void QueueEntryRemoval(Int row, Int column);

  // All queued entry additions and removals are applied, the entry list is
  // lexicographically sorted (the entries with the same locations are summed)
  // and the row offsets are updated.
  void FlushEntryQueues();

  // Returns true if there are no entries queued for addition or removal.
  bool EntryQueuesAreEmpty() const CATAMARI_NOEXCEPT;

  // Adds the entry (row, column, value) into the matrix.
  //
  // NOTE: This routine involves a merge sort involving all of the entries. It
  // is preferable to amortize this cost by batching together several entry
  // additions.
  void AddEntry(Int row, Int column, const Field& value);

  // Removed the entry (row, column) from the matrix.
  //
  // NOTE: This routine involves a merge sort involving all of the entries. It
  // is preferable to amortize this cost by batching together several entry
  // removals.
  void RemoveEntry(Int row, Int column);

  // Returns a reference to the entry with the given index.
  const MatrixEntry<Field>& Entry(Int entry_index) const CATAMARI_NOEXCEPT;

  // Returns a reference to the underlying vector of entries.
  // NOTE: Only the values are meant to be directly modified.
  Buffer<MatrixEntry<Field>>& Entries() CATAMARI_NOEXCEPT;

  // Returns a reference to the underlying vector of entries.
  const Buffer<MatrixEntry<Field>>& Entries() const CATAMARI_NOEXCEPT;

  // Returns the offset into the entry vector where entries from the given row
  // begin.
  Int RowEntryOffset(Int row) const CATAMARI_NOEXCEPT;

  // Returns the offset into the entry vector where the (row, column) entry
  // would be inserted.
  Int EntryOffset(Int row, Int column) const CATAMARI_NOEXCEPT;

  // Returns true if there is an entry at position (row, column).
  bool EntryExists(Int row, Int column) const CATAMARI_NOEXCEPT;

  // Returns the number of columns where the given row has entries.
  Int NumRowNonzeros(Int row) const CATAMARI_NOEXCEPT;

  // Returns a CoordinateGraph representing the nonzero pattern of the sparse
  // matrix.
  std::unique_ptr<quotient::CoordinateGraph> CoordinateGraph() const
      CATAMARI_NOEXCEPT;

 private:
  // The height of the matrix.
  Int num_rows_;

  // The width of the matrix.
  Int num_columns_;

  // The (lexicographically sorted) list of entries in the sparse matrix.
  // TODO(Jack Poulson): Avoid unnecessary traversals of this array due to
  // MatrixEntry not satisfying std::is_trivially_copy_constructible.
  Buffer<MatrixEntry<Field>> entries_;

  // A list of length 'num_rows_ + 1', where 'row_entry_offsets_[row]' indicates
  // the location in 'entries_' where an entry with indices (row, column) would
  // be inserted (ignoring any sorting based upon the value).
  Buffer<Int> row_entry_offsets_;

  // The list of entries currently queued for addition into the sparse matrix.
  std::vector<MatrixEntry<Field>> entries_to_add_;

  // The list of entries currently queued for removal from the sparse matrix.
  std::vector<GraphEdge> entries_to_remove_;

  // Incorporates the entries currently residing in 'entries_to_add_' into the
  // sparse matrix (and then clears 'entries_to_add_').
  void FlushEntryAdditionQueue(bool update_row_entry_offsets);

  // Removes the entries residing in 'entries_to_remove_' from the sparse matrix
  // (and then clears 'entries_to_remove_').
  void FlushEntryRemovalQueue(bool update_row_entry_offsets);

  // Recomputes 'row_entry_offsets_' based upon the current value of 'entries_'.
  void UpdateRowEntryOffsets();

  // Packs a sorted list of entries by summing the floating-point values of
  // entries with the same indices.
  static void CombineSortedEntries(std::vector<MatrixEntry<Field>>* entries);
  static void CombineSortedEntries(Buffer<MatrixEntry<Field>>* entries);
};

// Pretty-prints the CoordinateMatrix.
template <class Field>
void PrintCoordinateMatrix(const CoordinateMatrix<Field>& matrix,
                           const std::string& label);

}  // namespace catamari

#include "catamari/coordinate_matrix-impl.hpp"

#endif  // ifndef CATAMARI_COORDINATE_MATRIX_H_

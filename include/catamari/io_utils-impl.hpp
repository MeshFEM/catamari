/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_IO_UTILS_IMPL_H_
#define CATAMARI_IO_UTILS_IMPL_H_

#include <ostream>

#include "catamari/io_utils.hpp"

namespace catamari {

inline void TruncatedNodeLabelRecursion(Int supernode,
                                        const Buffer<quotient::Timer>& timers,
                                        const AssemblyForest& forest, Int level,
                                        Int max_levels, std::ofstream& file) {
  if (level > max_levels) {
    return;
  }

  // Write out this node's label.
  file << "  " << supernode << " [label=\"" << timers[supernode].TotalSeconds()
       << "\"]\n";

  const Int child_beg = forest.child_offsets[supernode];
  const Int child_end = forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = forest.children[child_beg + child_index];
    TruncatedNodeLabelRecursion(child, timers, forest, level + 1, max_levels,
                                file);
  }
}

inline void TruncatedNodeEdgeRecursion(Int supernode,
                                       const Buffer<quotient::Timer>& timers,
                                       const AssemblyForest& forest, Int level,
                                       Int max_levels, std::ofstream& file) {
  if (level >= max_levels) {
    return;
  }

  // Write out the child edge connections.
  const Int child_beg = forest.child_offsets[supernode];
  const Int child_end = forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = forest.children[child_beg + child_index];
    file << "  " << supernode << " -> " << child << "\n";
    TruncatedNodeEdgeRecursion(child, timers, forest, level + 1, max_levels,
                               file);
  }
}

inline void TruncatedForestTimersToDot(const std::string& filename,
                                       const Buffer<quotient::Timer>& timers,
                                       const AssemblyForest& forest,
                                       Int max_levels,
                                       bool avoid_isolated_roots) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open " << filename << std::endl;
    return;
  }

  const Int num_roots = forest.roots.Size();

  // Write out the header.
  file << "digraph g {\n";

  // Write out the node labels.
  Int level = 0;
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = forest.roots[root_index];
    if (!avoid_isolated_roots ||
        forest.child_offsets[root] != forest.child_offsets[root + 1]) {
      TruncatedNodeLabelRecursion(root, timers, forest, level, max_levels,
                                  file);
    }
  }

  file << "\n";

  // Write out the (truncated) connectivity.
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = forest.roots[root_index];
    if (!avoid_isolated_roots ||
        forest.child_offsets[root] != forest.child_offsets[root + 1]) {
      TruncatedNodeEdgeRecursion(root, timers, forest, level, max_levels, file);
    }
  }

  // Write out the footer.
  file << "}\n";
}

}  // namespace catamari

#endif  // ifndef CATAMARI_IO_UTILS_IMPL_H_

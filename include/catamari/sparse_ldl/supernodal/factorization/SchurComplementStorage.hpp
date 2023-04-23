////////////////////////////////////////////////////////////////////////////////
// SchurComplementStorage.hpp
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Manage the storage needed when computing the Schur complement blocks
// for a subtree of the elimination forest *serially* (as we do at the lowest
// levels). Our goal is to minimize calls to `malloc`. The
// original catamari code executed allocated memory separately for each
// supernode, which led to significant time spent in `malloc`, especially
// at the lower levels of the forest. We avoid this by allocating a single sufficiently
// large block of memory that we use as a matrix stack, pushing
// Schur complements as we compute them and then popping them as we merge them
// into their parent.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  04/19/2023 15:58:37
*///////////////////////////////////////////////////////////////////////////////
#ifndef SCHURCOMPLEMENTSTORAGE_HPP
#define SCHURCOMPLEMENTSTORAGE_HPP

#include "catamari/sparse_ldl/supernodal/factorization.hpp"
#include <Eigen/Dense>

namespace catamari {
namespace supernodal_ldl {

// Manage the storage used to evaluate the Schur complements of all
// *children* of a supernode in serial.
// (The subtree root supernode's Schur complement will be stored
// on the heap by the parent "parallel" node).
template<class Field>
struct SchurComplementStorage {
    // Calculate the total storage needed to evaluate the Schur complement block of
    // `supernode`'s frontal matrix (onto the top of the stack)
    static Int storageNeeded(Int supernode, const AssemblyForest &af, const LowerFactor<Field> &lf) {
        const Int degree = lf.blocks[supernode].height;
        if (af.NumChildren(supernode) == 0) {
            // At the leaves, we store only a single Schur complement
            return degree * degree;
        }
        const Int child_beg = af.child_offsets[supernode];
        const Int child_end = af.child_offsets[supernode + 1];

        // We use a rather primitive strategy of merging the child Schur complement
        // into the parent immediately after computing it. This avoids the need to store
        // the Schur complements of all children on the stack. However, this is not optimal
        // as it can force storage of Schur complements at multiple levels of the tree.
        Int maxStorage = storageNeeded(af.children[child_beg], af, lf) + degree * degree; // TODO: we can reduce this to `max(storageNeeded(child_beg), degree * degree)` by expanding the first child "in place"

        for (Int child_index = child_beg + 1; child_index < child_end; ++child_index) {
            const Int child = af.children[child_index];
            maxStorage = std::max(maxStorage, storageNeeded(child, af, lf) + degree * degree);
        }

        return maxStorage;
    }

    static Int storageNeededForChildren(Int supernode, const AssemblyForest &af, const LowerFactor<Field> &lf) {
        const Int child_beg = af.child_offsets[supernode];
        const Int child_end = af.child_offsets[supernode + 1];
        Int maxChildStorage = 0;
        for (Int child_index = child_beg; child_index < child_end; ++child_index) {
            const Int child = af.children[child_index];
            maxChildStorage = std::max(maxChildStorage, storageNeeded(child, af, lf));
        }
        return maxChildStorage;
    }

    SchurComplementStorage(Int cap = 0) { reserve(cap); }
    SchurComplementStorage(Int supernode, const AssemblyForest &af, const LowerFactor<Field> &lf) {
        reserve(storageNeededForChildren(supernode, af, lf));
    }

    void reserve(Int s) { m_storage.resize(s); }
    Int capacity() const { return m_storage.size(); }
    Int     size() const { return m_stackTop; }
    void deallocate() { m_storage.resize(0); }

    // Allocate a `n x n` matrix at the top of the stack
    BlasMatrixView<Field> push(Int n) {
        // if (size() + n * n > capacity()) throw std::runtime_error("Ran out of stack space attempting push " + std::to_string(n * n) + " at size " + std::to_string(size()) + "/" + std::to_string(capacity()));
        BlasMatrixView<Field> result;
        result.width = result.height = result.leading_dim = n;
        result.data = m_storage.data() + m_stackTop;
        m_stackTop += n * n;
        // std::cout << "Push " << n * n << ", new size " << size() << "/" << capacity() << std::endl;
        return result;
    }

    // Remove a `n x n` matrix from the top of the stack
    void pop(Int n) {
        // std::cout << "Pop " << n * n << " attempted at size " << size() << std::endl;
        // if (n * n > size()) throw std::runtime_error("Out-of-bounds pop");
        m_stackTop -= n * n;
    }

    void free(BlasMatrixView<Field> &sc) {
        pop(sc.height);
        sc.width = sc.height = 0;
        sc.data = nullptr;
    }

private:
    Eigen::Matrix<Field, Eigen::Dynamic, 1> m_storage;
    Int m_stackTop = 0;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#endif /* end of include guard: SCHURCOMPLEMENTSTORAGE_HPP */

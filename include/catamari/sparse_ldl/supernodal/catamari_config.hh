#ifndef CATAMARI_CONFIG_HH
#define CATAMARI_CONFIG_HH

// Control various optimizations/tradeoffs...
#define LOAD_MATRIX_OUTSIDE 1

// On-the-fly construction of the Schur complement buffers
// (setting to 1 obtains the original implementation, which is slightly slower for small matrices but saves memory).
#define ALLOCATE_SCHUR_COMPLEMENT_OTF 1

#define CUSTOM_TIMERS 0

#endif /* end of include guard: CATAMARI_CONFIG_HH */

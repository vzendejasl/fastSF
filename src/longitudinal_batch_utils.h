#ifndef LONGITUDINAL_BATCH_UTILS_H
#define LONGITUDINAL_BATCH_UTILS_H

#include <blitz/array.h>
#include <vector>

int packed_longitudinal_3d_value_index(int ix, int z, int order_index, int nz_half, int order_count);

void unpack_longitudinal_3d_gathered_values(
    const std::vector<double>& gathered_values,
    const blitz::Array<int,3>& idx,
    int rank_count,
    int c_per_proc,
    int nz_half,
    int order_count,
    blitz::Array<double,4>& sf_grid);

#endif

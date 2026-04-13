#include "longitudinal_batch_utils.h"

#include <blitz/array.h>
#include <cassert>
#include <iostream>
#include <vector>

namespace {

void expect_packed_indices() {
    assert(packed_longitudinal_3d_value_index(0, 0, 0, 4, 2) == 0);
    assert(packed_longitudinal_3d_value_index(0, 0, 1, 4, 2) == 1);
    assert(packed_longitudinal_3d_value_index(0, 1, 0, 4, 2) == 2);
    assert(packed_longitudinal_3d_value_index(1, 0, 0, 4, 2) == 8);
    assert(packed_longitudinal_3d_value_index(1, 3, 1, 4, 2) == 15);
}

void expect_unpack_layout() {
    const int rank_count = 2;
    const int c_per_proc = 2;
    const int nz_half = 3;
    const int order_count = 2;

    blitz::Array<int,3> idx(c_per_proc, 2, rank_count);
    idx = -1;
    idx(0,0,0) = 0; idx(0,1,0) = 0;
    idx(1,0,0) = 0; idx(1,1,0) = 1;
    idx(0,0,1) = 1; idx(0,1,1) = 0;
    idx(1,0,1) = 1; idx(1,1,1) = 1;

    std::vector<double> gathered(rank_count * c_per_proc * nz_half * order_count, -1.0);
    for (int rank_id = 0; rank_id < rank_count; rank_id++) {
        int rank_base = rank_id * c_per_proc * nz_half * order_count;
        for (int ix = 0; ix < c_per_proc; ix++) {
            for (int z = 0; z < nz_half; z++) {
                for (int order_index = 0; order_index < order_count; order_index++) {
                    int value_index = rank_base + packed_longitudinal_3d_value_index(ix, z, order_index, nz_half, order_count);
                    gathered[value_index] = 1000.0 * rank_id + 100.0 * ix + 10.0 * z + order_index;
                }
            }
        }
    }

    blitz::Array<double,4> sf_grid(2, 2, nz_half, order_count);
    sf_grid = -99.0;
    unpack_longitudinal_3d_gathered_values(gathered, idx, rank_count, c_per_proc, nz_half, order_count, sf_grid);

    assert(sf_grid(0,0,0,0) == 0.0);
    assert(sf_grid(0,0,2,1) == 21.0);
    assert(sf_grid(0,1,1,0) == 110.0);
    assert(sf_grid(1,0,2,1) == 1021.0);
    assert(sf_grid(1,1,0,0) == 1100.0);
    assert(sf_grid(1,1,2,1) == 1121.0);
}

}  // namespace

int main() {
    expect_packed_indices();
    expect_unpack_layout();
    std::cout << "longitudinal batch utils tests passed\n";
    return 0;
}

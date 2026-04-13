#include "longitudinal_batch_utils.h"

#include <stdexcept>

int packed_longitudinal_3d_value_index(int ix, int z, int order_index, int nz_half, int order_count) {
    return ((ix * nz_half) + z) * order_count + order_index;
}

void unpack_longitudinal_3d_gathered_values(
    const std::vector<double>& gathered_values,
    const blitz::Array<int,3>& idx,
    int rank_count,
    int c_per_proc,
    int nz_half,
    int order_count,
    blitz::Array<double,4>& sf_grid) {
    int values_per_rank = c_per_proc * nz_half * order_count;
    if (static_cast<int>(gathered_values.size()) != rank_count * values_per_rank) {
        throw std::runtime_error("unexpected gathered value count for 3D longitudinal unpack");
    }

    for (int rank_id = 0; rank_id < rank_count; rank_id++) {
        int rank_base = rank_id * values_per_rank;
        for (int ix = 0; ix < c_per_proc; ix++) {
            int x = idx(ix, 0, rank_id);
            int y = idx(ix, 1, rank_id);
            for (int z = 0; z < nz_half; z++) {
                int value_base = rank_base + packed_longitudinal_3d_value_index(ix, z, 0, nz_half, order_count);
                for (int order_index = 0; order_index < order_count; order_index++) {
                    sf_grid(x, y, z, order_index) = gathered_values[value_base + order_index];
                }
            }
        }
    }
}

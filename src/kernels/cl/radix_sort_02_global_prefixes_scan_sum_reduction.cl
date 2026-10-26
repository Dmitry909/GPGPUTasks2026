#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global uint* output,
    __global uint* block_sums,
    unsigned int n)
{
    const uint local_index = get_local_id(0);
    const uint index = get_global_id(0);
    __local uint values[GROUP_SIZE];
    if (index < n) {
        values[local_index] = input[index];
    } else {
        values[local_index] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pow = 1; pow < GROUP_SIZE; pow <<= 1) {
        uint mul = 2 * pow;
        uint right = (local_index + 1) * mul - 1;

        if (right < GROUP_SIZE) {
            values[right] += values[right - pow];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        block_sums[get_group_id(0)] = values[GROUP_SIZE - 1];
    }

    for (uint pow = GROUP_SIZE / 4; pow > 0; pow >>= 1) {
        uint mul = 2 * pow;
        uint right = (local_index + 1) * mul - 1;

        if (right + pow < GROUP_SIZE) {
            values[right + pow] += values[right];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index < n) {
        output[index] = values[local_index];
    }
}

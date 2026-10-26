#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* output,
    unsigned int n,
    unsigned int bit_offset)
{
    const size_t global_index = get_global_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_counts[16];
    if (local_index < 16) {
        local_counts[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_index < n) {
        uint value = (input[global_index] >> bit_offset) & 0b1111;
        atomic_inc(&local_counts[value]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < 16) {
        output[local_index * get_num_groups(0) + get_group_id(0)] = local_counts[local_index];
    }
}

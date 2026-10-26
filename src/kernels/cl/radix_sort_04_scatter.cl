#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sum,
              __global uint* result,
    unsigned int n,
    unsigned int bit_offset)
{
    const size_t global_index = get_global_id(0);
    const uint local_index = get_local_id(0);

    __local uint digits[GROUP_SIZE];
    uint current = 0;
    uint bits = 16;
    if (global_index < n) {
        current = input[global_index];
        bits = (current >> bit_offset) & 0b1111;
    }
    digits[local_index] = bits;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_index < n) {
        uint counter_index = bits * get_num_groups(0) + get_group_id(0);
        uint result_index = 0;
        if (counter_index > 0) {
            result_index = prefix_sum[counter_index - 1];
        }
        for (uint i = 0; i < local_index; ++i) {
            if (digits[i] == bits) {
                ++result_index;
            }
        }

        result[result_index] = current;
    }
}
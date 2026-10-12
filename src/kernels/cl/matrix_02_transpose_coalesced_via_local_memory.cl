#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    __local float local_group[GROUP_SIZE_Y][GROUP_SIZE_X];
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    if (x < w && y < h) {
        local_group[local_y][local_x] = matrix[x + y * w];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint result_x = y / GROUP_SIZE_Y * GROUP_SIZE_Y + local_x;
    const uint result_y = x / GROUP_SIZE_X * GROUP_SIZE_X + local_y;

    if (result_x < h && result_y < w) {
        transposed_matrix[result_x + result_y * h] = local_group[local_x][local_y];
    }
}

#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

#define STEP GROUP_SIZE_X

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    __local float rows[GROUP_SIZE_Y][STEP];
    __local float columns[GROUP_SIZE_X][STEP];
    float result = 0;

    const size_t step_count = (k + STEP - 1) / STEP;
    for (size_t step_index = 0; step_index < step_count; ++step_index) {
        const size_t offset = step_index * STEP;

        if (y < h && offset + local_x < k) {
            rows[local_y][local_x] = a[y * k + offset + local_x];
        } else {
            rows[local_y][local_x] = 0;
        }

        if (x < w && offset + local_y < k) {
            columns[local_x][local_y] = b[(offset + local_y) * w + x];
        } else {
            columns[local_x][local_y] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < STEP; ++i) {
            result += rows[local_y][i] * columns[local_x][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = result;
    }
}

enable f16;
enable subgroups;

#if defined(MUL_MAT_VEC_F32_F32_VEC)
#define SRC0_TYPE vec4<f32>
#define SRC1_TYPE vec4<f32>
#define DST_TYPE vec4<f32>
#define VEC_SIZE 4u
#define VEC
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_F32_F32)
#define SRC0_TYPE f32
#define SRC1_TYPE f32
#define DST_TYPE f32
#define VEC_SIZE 1u
#define SCALAR
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_F16_F32_VEC)
#define SRC0_TYPE vec4<f16>
#define SRC1_TYPE vec4<f32>
#define DST_TYPE vec4<f32>
#define VEC_SIZE 4u
#define VEC
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_F16_F32)
#define SRC0_TYPE f16
#define SRC1_TYPE f32
#define DST_TYPE f32
#define VEC_SIZE 1u
#define SCALAR
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_F16_F16_VEC)
#define SRC0_TYPE vec4<f16>
#define SRC1_TYPE vec4<f16>
#define DST_TYPE vec4<f32>
#define VEC_SIZE 4u
#define VEC
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_F16_F16)
#define SRC0_TYPE f16
#define SRC1_TYPE f16
#define DST_TYPE f32
#define VEC_SIZE 1u
#define SCALAR
#define MUL_ACC_FLOAT
#elif defined(MUL_MAT_VEC_Q4_0_F32)
#define SRC0_TYPE f16
#define SRC1_TYPE f32
#define DST_TYPE f32
#define VEC_SIZE 1u
#define SCALAR
#define BYTE_HELPERS
#define MUL_ACC_Q4_0
#endif

#if defined(BYTE_HELPERS)
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8u)) & 0xFFu;
}
#endif

#if defined(VEC)
fn inner_mul(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> vec4<f32> {
    return vec4<f32>(src0_val) * vec4<f32>(src1_val);
}

fn reduce_vec4(v: vec4<f32>) -> f32 {
    return v.x + v.y + v.z + v.w;
}

fn store_val(dst_idx: u32, dst: ptr<storage, array<DST_TYPE>, read_write>, subgroup_invocation_id: u32, subgroup_size: u32, num_subgroups: u32, row_base: u32) {
    let lane = subgroup_invocation_id;
    for (var row = 0u; row < ROWS_PER_WG && row_base + row + VEC_SIZE - 1u < params.m; row += VEC_SIZE) {
        let v0 = select(0.0, partial_sums[(row + 0u) * MAX_SUBGROUP_SIZE + lane], lane < num_subgroups);
        let v1 = select(0.0, partial_sums[(row + 1u) * MAX_SUBGROUP_SIZE + lane], lane < num_subgroups);
        let v2 = select(0.0, partial_sums[(row + 2u) * MAX_SUBGROUP_SIZE + lane], lane < num_subgroups);
        let v3 = select(0.0, partial_sums[(row + 3u) * MAX_SUBGROUP_SIZE + lane], lane < num_subgroups);

        let vec_tot = vec4<f32>(subgroupAdd(v0), subgroupAdd(v1), subgroupAdd(v2), subgroupAdd(v3));

        if (subgroup_invocation_id == 0u) {
            (*dst)[(dst_idx + row) / VEC_SIZE] = vec_tot;
        }
    }
}
#endif

#if defined(SCALAR)
fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

#if defined(MUL_ACC_Q4_0)
const BLOCK_SIZE = 32u;
const F16_PER_BLOCK = 9u; // 1 scale + 8 packed words (2 bytes each)

fn q4_0_mul(idx_base: u32, k_idx: u32, b: f32) -> f32 {
    let block_idx = idx_base + (k_idx / BLOCK_SIZE);
    let scale_idx = block_idx * F16_PER_BLOCK;
    let d = f32(src0[scale_idx]);

    let block_offset = k_idx % BLOCK_SIZE;
    let byte_idx = block_offset % 16u;
    let q_word_idx = scale_idx + 1u + (byte_idx / 2u);
    let q_word = bitcast<u32>(vec2(src0[q_word_idx], src0[q_word_idx]));
    let q_byte = get_byte(q_word, byte_idx % 2u);
    let q_u = select((q_byte >> 4u) & 0xFu, q_byte & 0xFu, block_offset < 16u);

    return (f32(q_u) - 8.0) * d * b;
}
#endif

fn store_val(dst_idx: u32, dst: ptr<storage, array<DST_TYPE>, read_write>, subgroup_invocation_id: u32, subgroup_size: u32, num_subgroups: u32, row_base: u32) {
    let lane = subgroup_invocation_id;
    for (var row = 0u; row < ROWS_PER_WG && row_base + row < params.m; row++) {
        let v = select(0.0, partial_sums[row * MAX_SUBGROUP_SIZE + lane], lane < num_subgroups);
        let tot = subgroupAdd(v);

        if (subgroup_invocation_id == 0u) {
            (*dst)[dst_idx + row] = tot;
        }
    }
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // Matrix (M x K)
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // Vector (K x 1, transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;  // Result vector (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

override WORKGROUP_SIZE: u32;
override OUTPUTS_PER_WG: u32;

const TILE_K = 4u / VEC_SIZE; // 4
const ROWS_PER_WG = 16u; // 16
override MAX_SUBGROUP_SIZE: u32;

// Shared memory for collaborative loading and reduction
var<workgroup> partial_sums:array<f32, MAX_SUBGROUP_SIZE*OUTPUTS_PER_WG>;   // For reduction

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(num_subgroups) num_subgroups: u32) {

    // Handle batch dimensions
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    // more portable than local_id.x
    let thread_id = subgroup_invocation_id + subgroup_id * subgroup_size;

    let k = params.k / VEC_SIZE;
    let row_base = (wg_linear % output_groups) * OUTPUTS_PER_WG;


    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    // physical dst index is params.offset_dst - but offset by batch and even further so the output row
    let src1_idx_base = (params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12) / VEC_SIZE;
    let dst_idx = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row_base;

    var tile: array<SRC1_TYPE, TILE_K>;
    var row_indeces: array<u32, ROWS_PER_WG>;
    var sumf: array<f32, ROWS_PER_WG> = array<f32, ROWS_PER_WG>();


    for (var row = 0u; row < ROWS_PER_WG; row++) {
        row_indeces[row] = (params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + (row_base + row) * params.stride_01) / VEC_SIZE;
    }

    let subgroup_base = subgroup_id * subgroup_size * TILE_K;

    let stride = WORKGROUP_SIZE * TILE_K;

    var ib = subgroup_base;

    while (ib + stride <= k) {

        // load from B vector
        for (var i = 0u; i < TILE_K; i++) {
            tile[i] = src1[src1_idx_base + ib + (i * subgroup_size) + subgroup_invocation_id];
        }

        for (var row = 0u; row < ROWS_PER_WG; row++) {

            let my_id = row_indeces[row] + ib + subgroup_invocation_id;

#if defined(VEC)
            var sumqv = vec4<f32>(0.0);

            // load from A and register tiled B
            for (var i = 0u; i < TILE_K; i++) {
                sumqv += inner_mul(src0[my_id + (i * subgroup_size)], tile[i]);
            }
            sumf[row] += reduce_vec4(sumqv);
#elif defined(SCALAR)
            var sumq = 0.0;

            // load from A and register tiled B
            for (var i = 0u; i < TILE_K; i++) {
#if defined(MUL_ACC_Q4_0)
                let k_idx = ib + (i * subgroup_size) + subgroup_invocation_id;
                sumq += q4_0_mul(row_indeces[row], k_idx, tile[i]);
#else
                sumq += inner_dot(src0[my_id + (i * subgroup_size)], tile[i]);
#endif
            }
            sumf[row] += sumq;
#endif
        }
        ib += stride;
    }


    // tail
    // load from B vector
    if (ib < k) {
        for (var i = 0u; i < TILE_K; i++) {
            let k_idx = ib + (i * subgroup_size) + subgroup_invocation_id;
            if (k_idx < k) {
                tile[i] = src1[src1_idx_base + k_idx];
            }
        }
    }

    for (var row = 0u; row < ROWS_PER_WG; row++) {

        let my_id = row_indeces[row] + ib + subgroup_invocation_id;

#if defined(VEC)
        var sumqv = vec4<f32>(0.0);

        // load from A and register tiled B
        for (var i = 0u; i < TILE_K; i++) {
            let k_idx = ib + (i * subgroup_size) + subgroup_invocation_id;
            if (k_idx < k) {
                sumqv += inner_mul(src0[my_id + (i * subgroup_size)], tile[i]);
            }

        }
        sumf[row] += reduce_vec4(sumqv);
#elif defined(SCALAR)
        var sumq = 0.0;

        // load from A and register tiled B
        for (var i = 0u; i < TILE_K; i++) {
            let k_idx = ib + (i * subgroup_size) + subgroup_invocation_id;
            if (k_idx < k) {
#if defined(MUL_ACC_Q4_0)
                sumq += q4_0_mul(row_indeces[row], k_idx, tile[i]);
#else
                sumq += inner_dot(src0[my_id + (i * subgroup_size)], tile[i]);
#endif
            }

        }
        sumf[row] += sumq;
#endif
    }


    let fast_subgroup_only = (WORKGROUP_SIZE == subgroup_size);
    if (fast_subgroup_only) {
#if defined(VEC)
        for (var row = 0u; row < ROWS_PER_WG && row_base + row + VEC_SIZE - 1u < params.m; row += VEC_SIZE) {
            let v0 = subgroupAdd(sumf[row + 0u]);
            let v1 = subgroupAdd(sumf[row + 1u]);
            let v2 = subgroupAdd(sumf[row + 2u]);
            let v3 = subgroupAdd(sumf[row + 3u]);
            if (subgroup_invocation_id == 0u) {
                dst[(dst_idx + row) / VEC_SIZE] = vec4<f32>(v0, v1, v2, v3);
            }
        }
#endif
#if defined(SCALAR)
        for (var row = 0u; row < ROWS_PER_WG && row_base + row < params.m; row++) {
            let tot = subgroupAdd(sumf[row]);
            if (subgroup_invocation_id == 0u) {
                dst[dst_idx + row] = tot;
            }
        }
#endif
    } else {
        // Subgroup-size-agnostic reduction:
        for (var row = 0u; row < ROWS_PER_WG; row++) {
            sumf[row] = subgroupAdd(sumf[row]);
            if (subgroup_invocation_id == 0u) {
                partial_sums[row * MAX_SUBGROUP_SIZE + subgroup_id] = sumf[row];
            }
        }

        workgroupBarrier();

        if (subgroup_id == 0u) {
            store_val(dst_idx, &dst, subgroup_invocation_id, subgroup_size, num_subgroups, row_base);
        }
    }

}

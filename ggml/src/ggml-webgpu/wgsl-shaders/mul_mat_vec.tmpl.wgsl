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
        var v0 = 0.0;
        var v1 = 0.0;
        var v2 = 0.0;
        var v3 = 0.0;
        for (var sg = lane; sg < num_subgroups; sg += subgroup_size) {
            v0 += partial_sums[(row + 0u) * WORKGROUP_SIZE + sg];
            v1 += partial_sums[(row + 1u) * WORKGROUP_SIZE + sg];
            v2 += partial_sums[(row + 2u) * WORKGROUP_SIZE + sg];
            v3 += partial_sums[(row + 3u) * WORKGROUP_SIZE + sg];
        }
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
        var v = 0.0;
        for (var sg = lane; sg < num_subgroups; sg += subgroup_size) {
            v += partial_sums[row * WORKGROUP_SIZE + sg];
        }
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
const ROWS_PER_WG = OUTPUTS_PER_WG;

// Shared memory for subgroup results before final workgroup reduction.
var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE * OUTPUTS_PER_WG>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(num_subgroups) num_subgroups: u32) {

    // Handle batch dimensions.
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + ROWS_PER_WG - 1u) / ROWS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    let k = params.k / VEC_SIZE;
    let row_base = (wg_linear % output_groups) * ROWS_PER_WG;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src1_idx_base = (params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12) / VEC_SIZE;
    let dst_idx = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row_base;

    var sumf: array<f32, OUTPUTS_PER_WG> = array<f32, OUTPUTS_PER_WG>();

    // Each subgroup starts at its own offset within a workgroup-wide K stripe.
    let subgroup_base = subgroup_id * subgroup_size;
    let stride = WORKGROUP_SIZE;

    var ib = subgroup_base;
    while (ib < k) {
        // Each lane owns one source1 element and serves it to the subgroup via broadcast.
        let lane_k = ib + subgroup_invocation_id;
        var lane_b = SRC1_TYPE(0.0);
        if (lane_k < k) {
            lane_b = src1[src1_idx_base + lane_k];
        }

        for (var j = 0u; j < subgroup_size && ib + j < k; j++) {
            let b = subgroupBroadcast(lane_b, j);
            let k_idx = ib + j;

            for (var row = subgroup_invocation_id; row < ROWS_PER_WG && row_base + row < params.m; row += subgroup_size) {
                let row_idx = (params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + (row_base + row) * params.stride_01) / VEC_SIZE;
#if defined(VEC)
                sumf[row] += reduce_vec4(inner_mul(src0[row_idx + k_idx], b));
#elif defined(MUL_ACC_Q4_0)
                sumf[row] += q4_0_mul(row_idx, k_idx, b);
#else
                sumf[row] += inner_dot(src0[row_idx + k_idx], b);
#endif
            }
        }

        ib += stride;
    }



    // First reduce per-row inside each subgroup.
    for (var row = 0u; row < ROWS_PER_WG; row++) {
        sumf[row] = subgroupAdd(sumf[row]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[row * WORKGROUP_SIZE + subgroup_id] = sumf[row];
        }
    }

    workgroupBarrier();

    // Then have subgroup 0 reduce subgroup totals and write results.
    if (subgroup_id == 0u) {
        store_val(dst_idx, &dst, subgroup_invocation_id, subgroup_size, num_subgroups, row_base);
    }
}

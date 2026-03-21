// SIMD-optimized int8 neural network inference for aarch64 (Apple Silicon).
//
// Uses NEON intrinsics with dotprod extension for fast int8 matrix multiply.
// Key optimizations:
// - Int8 quantized weights and activations
// - vdotq_s32 for 4-element int8 dot products
// - Fused BatchNorm (folded into conv weights at export time)
// - NHWC layout not used yet — sticking with NCHW for initial impl

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Quantization parameters for a single tensor.
#[derive(Clone, Debug)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i8,
}

/// Quantize f32 tensor to int8 with symmetric quantization.
/// Returns (quantized_data, scale).
pub fn quantize_symmetric(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }

    let abs_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max < 1e-10 { 1.0 } else { abs_max / 127.0 };
    let inv_scale = 1.0 / scale;

    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let q = (x * inv_scale).round() as i32;
            q.clamp(-127, 127) as i8
        })
        .collect();

    (quantized, scale)
}

/// Dequantize int8 back to f32.
pub fn dequantize(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x as f32 * scale).collect()
}

/// Int8 matrix-vector multiply: output[i] = sum_j(weight[i][j] * input[j])
///
/// weight: [out_features, in_features] in int8
/// input: [in_features] in int8
/// output: [out_features] in int32 (accumulator)
///
/// On aarch64, uses vmull_s8/vmlal_s8 to widen int8 -> int16 products,
/// then vpadalq_s16 to accumulate int16 pairs -> int32.
/// Processes 8 int8 multiplies per vmull/vmlal instruction.
#[cfg(target_arch = "aarch64")]
pub fn matvec_int8_neon(
    weight: &[i8],
    input: &[i8],
    output: &mut [i32],
    out_features: usize,
    in_features: usize,
) {
    assert_eq!(weight.len(), out_features * in_features);
    assert_eq!(input.len(), in_features);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        unsafe {
            let mut acc = vdupq_n_s32(0);

            // Process 8 elements at a time using vmull_s8 (8xi8 * 8xi8 -> 8xi16)
            let chunks = in_features / 8;

            for c in 0..chunks {
                let offset = c * 8;
                let inp = vld1_s8(input.as_ptr().add(offset));
                let w = vld1_s8(weight.as_ptr().add(row * in_features + offset));

                // Multiply 8xi8 -> 8xi16
                let prod = vmull_s8(w, inp);
                // Pairwise add 8xi16 -> 4xi32 and accumulate
                acc = vpadalq_s16(acc, prod);
            }

            // Horizontal sum of 4xi32 -> scalar
            output[row] = vaddvq_s32(acc);

            // Handle remaining elements
            for j in (chunks * 8)..in_features {
                output[row] += weight[row * in_features + j] as i32 * input[j] as i32;
            }
        }
    }
}

/// Fallback scalar implementation for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
pub fn matvec_int8_neon(
    weight: &[i8],
    input: &[i8],
    output: &mut [i32],
    out_features: usize,
    in_features: usize,
) {
    for i in 0..out_features {
        let mut sum = 0i32;
        for j in 0..in_features {
            sum += weight[i * in_features + j] as i32 * input[j] as i32;
        }
        output[i] = sum;
    }
}

/// Int8 3x3 convolution for a single output channel.
///
/// input: [in_c, H, W] in int8
/// kernel: [in_c, 3, 3] in int8
/// output: [H, W] in int32 (accumulator, includes padding=1)
#[cfg(target_arch = "aarch64")]
pub fn conv3x3_int8_neon(
    input: &[i8],
    kernel: &[i8],
    output: &mut [i32],
    in_channels: usize,
    height: usize,
    width: usize,
) {
    assert_eq!(kernel.len(), in_channels * 9);
    let hw = height * width;
    let total_elems = in_channels * 9;
    let mut gather_buf = vec![0i8; total_elems];

    // For each output position
    for oh in 0..height {
        for ow in 0..width {
            unsafe {
                let mut acc = vdupq_n_s32(0);
                let mut elem_idx = 0;

                // Gather input patch * kernel for all input channels

                for ic in 0..in_channels {
                    for kh in 0..3usize {
                        for kw in 0..3usize {
                            let ih = oh as isize + kh as isize - 1;
                            let iw = ow as isize + kw as isize - 1;

                            let val = if ih >= 0 && ih < height as isize && iw >= 0 && iw < width as isize {
                                input[ic * hw + ih as usize * width + iw as usize]
                            } else {
                                0i8 // zero padding
                            };
                            gather_buf[elem_idx] = val;
                            elem_idx += 1;
                        }
                    }
                }

                // NEON multiply-accumulate over gathered buffer and kernel
                let chunks = total_elems / 8;
                for c in 0..chunks {
                    let offset = c * 8;
                    let inp = vld1_s8(gather_buf.as_ptr().add(offset));
                    let ker = vld1_s8(kernel.as_ptr().add(offset));
                    let prod = vmull_s8(ker, inp);
                    acc = vpadalq_s16(acc, prod);
                }

                let mut result = vaddvq_s32(acc);

                // Remainder
                for j in (chunks * 16)..total_elems {
                    result += gather_buf[j] as i32 * kernel[j] as i32;
                }

                output[oh * width + ow] = result;
            }
        }
    }
}

/// Fallback scalar conv3x3.
#[cfg(not(target_arch = "aarch64"))]
pub fn conv3x3_int8_neon(
    input: &[i8],
    kernel: &[i8],
    output: &mut [i32],
    in_channels: usize,
    height: usize,
    width: usize,
) {
    let hw = height * width;
    for oh in 0..height {
        for ow in 0..width {
            let mut sum = 0i32;
            for ic in 0..in_channels {
                for kh in 0..3isize {
                    for kw in 0..3isize {
                        let ih = oh as isize + kh - 1;
                        let iw = ow as isize + kw - 1;
                        let val = if ih >= 0 && ih < height as isize && iw >= 0 && iw < width as isize {
                            input[ic * hw + ih as usize * width + iw as usize] as i32
                        } else {
                            0
                        };
                        let k = kernel[ic * 9 + kh as usize * 3 + kw as usize] as i32;
                        sum += val * k;
                    }
                }
            }
            output[oh * width + ow] = sum;
        }
    }
}

/// Quantized ReLU: clamp to [0, 127] for int8.
pub fn relu_int8(data: &mut [i8]) {
    for v in data.iter_mut() {
        if *v < 0 {
            *v = 0;
        }
    }
}

/// Full quantized network forward pass.
/// This is the high-performance inference path used during MCTS.
pub struct Int8Evaluator {
    // Quantized weights for each layer
    stem_weight: Vec<i8>,
    stem_weight_scale: f32,
    stem_bias: Vec<f32>,
    residual_weights: Vec<ResBlockInt8>,
    policy_conv_weight: Vec<i8>,
    policy_conv_scale: f32,
    policy_conv_bias: Vec<f32>,
    policy_fc_weight: Vec<i8>,
    policy_fc_scale: f32,
    policy_fc_bias: Vec<f32>,
    value_conv_weight: Vec<i8>,
    value_conv_scale: f32,
    value_conv_bias: Vec<f32>,
    value_fc1_weight: Vec<i8>,
    value_fc1_scale: f32,
    value_fc1_bias: Vec<f32>,
    value_fc2_weight: Vec<i8>,
    value_fc2_scale: f32,
    value_fc2_bias: Vec<f32>,
    num_filters: usize,
    _num_blocks: usize,
}

pub struct ResBlockInt8 {
    conv1_weight: Vec<i8>,
    conv1_scale: f32,
    conv1_bias: Vec<f32>,
    conv2_weight: Vec<i8>,
    conv2_scale: f32,
    conv2_bias: Vec<f32>,
}

impl Int8Evaluator {
    /// Create from f32 NetworkWeights by quantizing.
    pub fn from_weights(weights: &crate::nn::weights::NetworkWeights) -> Self {
        let (stem_w, stem_s) = quantize_symmetric(&weights.stem_conv_weight);
        let nf = weights.num_filters;

        let residual_weights: Vec<ResBlockInt8> = weights.residual_blocks.iter().map(|block| {
            let (c1w, c1s) = quantize_symmetric(&block.conv1_weight);
            let (c2w, c2s) = quantize_symmetric(&block.conv2_weight);
            ResBlockInt8 {
                conv1_weight: c1w,
                conv1_scale: c1s,
                conv1_bias: block.conv1_bias.clone(),
                conv2_weight: c2w,
                conv2_scale: c2s,
                conv2_bias: block.conv2_bias.clone(),
            }
        }).collect();

        let (pcw, pcs) = quantize_symmetric(&weights.policy_conv_weight);
        let (pfw, pfs) = quantize_symmetric(&weights.policy_fc_weight);
        let (vcw, vcs) = quantize_symmetric(&weights.value_conv_weight);
        let (vf1w, vf1s) = quantize_symmetric(&weights.value_fc1_weight);
        let (vf2w, vf2s) = quantize_symmetric(&weights.value_fc2_weight);

        Int8Evaluator {
            stem_weight: stem_w,
            stem_weight_scale: stem_s,
            stem_bias: weights.stem_conv_bias.clone(),
            residual_weights,
            policy_conv_weight: pcw,
            policy_conv_scale: pcs,
            policy_conv_bias: weights.policy_conv_bias.clone(),
            policy_fc_weight: pfw,
            policy_fc_scale: pfs,
            policy_fc_bias: weights.policy_fc_bias.clone(),
            value_conv_weight: vcw,
            value_conv_scale: vcs,
            value_conv_bias: weights.value_conv_bias.clone(),
            value_fc1_weight: vf1w,
            value_fc1_scale: vf1s,
            value_fc1_bias: weights.value_fc1_bias.clone(),
            value_fc2_weight: vf2w,
            value_fc2_scale: vf2s,
            value_fc2_bias: weights.value_fc2_bias.clone(),
            num_filters: nf,
            _num_blocks: weights.num_blocks,
        }
    }

    /// Forward pass with int8 convolutions and f32 accumulators.
    /// Uses a hybrid approach: int8 for compute-heavy conv/linear, f32 for activations.
    fn forward(&self, input: &[f32; crate::board::INPUT_SIZE]) -> crate::mcts::search::NnOutput {
        let nf = self.num_filters;
        let h = 8usize;
        let w = 8usize;
        let hw = h * w;

        // Quantize input
        let (input_q, input_scale) = quantize_symmetric(input);

        // Stem conv: int8 conv3x3, then dequantize + bias + relu
        let mut x_f32 = vec![0.0f32; nf * hw];
        for oc in 0..nf {
            let kernel = &self.stem_weight[oc * 21 * 9..(oc + 1) * 21 * 9];
            let mut out_i32 = vec![0i32; hw];
            conv3x3_int8_neon(&input_q, kernel, &mut out_i32, 21, h, w);

            let combined_scale = input_scale * self.stem_weight_scale;
            for i in 0..hw {
                let val = out_i32[i] as f32 * combined_scale + self.stem_bias[oc];
                x_f32[oc * hw + i] = val.max(0.0); // ReLU
            }
        }

        // Residual blocks
        for block in &self.residual_weights {
            let residual = x_f32.clone();

            // Conv1: quantize input, int8 conv, dequant + bias + relu
            let (x_q, x_scale) = quantize_symmetric(&x_f32);
            let mut conv1_out = vec![0.0f32; nf * hw];
            for oc in 0..nf {
                let kernel = &block.conv1_weight[oc * nf * 9..(oc + 1) * nf * 9];
                let mut out_i32 = vec![0i32; hw];
                conv3x3_int8_neon(&x_q, kernel, &mut out_i32, nf, h, w);

                let combined_scale = x_scale * block.conv1_scale;
                for i in 0..hw {
                    let val = out_i32[i] as f32 * combined_scale + block.conv1_bias[oc];
                    conv1_out[oc * hw + i] = val.max(0.0); // ReLU
                }
            }

            // Conv2: quantize conv1 output, int8 conv, dequant + bias + skip + relu
            let (c1_q, c1_scale) = quantize_symmetric(&conv1_out);
            for oc in 0..nf {
                let kernel = &block.conv2_weight[oc * nf * 9..(oc + 1) * nf * 9];
                let mut out_i32 = vec![0i32; hw];
                conv3x3_int8_neon(&c1_q, kernel, &mut out_i32, nf, h, w);

                let combined_scale = c1_scale * block.conv2_scale;
                for i in 0..hw {
                    let val = out_i32[i] as f32 * combined_scale + block.conv2_bias[oc];
                    x_f32[oc * hw + i] = (val + residual[oc * hw + i]).max(0.0); // skip + ReLU
                }
            }
        }

        // Policy head: 1x1 conv (as matmul per spatial position)
        let (x_q, x_scale) = quantize_symmetric(&x_f32);
        let mut policy_conv_out = vec![0.0f32; 32 * hw];
        for oc in 0..32 {
            for pos in 0..hw {
                let mut sum = 0i32;
                for ic in 0..nf {
                    sum += self.policy_conv_weight[oc * nf + ic] as i32 * x_q[ic * hw + pos] as i32;
                }
                let val = sum as f32 * (x_scale * self.policy_conv_scale) + self.policy_conv_bias[oc];
                policy_conv_out[oc * hw + pos] = val.max(0.0);
            }
        }

        // Policy FC
        let (pc_q, pc_scale) = quantize_symmetric(&policy_conv_out);
        let mut policy_i32 = vec![0i32; 1858];
        matvec_int8_neon(&self.policy_fc_weight, &pc_q, &mut policy_i32, 1858, 32 * hw);
        let policy: Vec<f32> = (0..1858)
            .map(|i| policy_i32[i] as f32 * (pc_scale * self.policy_fc_scale) + self.policy_fc_bias[i])
            .collect();

        // Value head: 1x1 conv
        let mut value_conv_out = vec![0.0f32; hw];
        for pos in 0..hw {
            let mut sum = 0i32;
            for ic in 0..nf {
                sum += self.value_conv_weight[ic] as i32 * x_q[ic * hw + pos] as i32;
            }
            let val = sum as f32 * (x_scale * self.value_conv_scale) + self.value_conv_bias[0];
            value_conv_out[pos] = val.max(0.0);
        }

        // Value FC1
        let (vc_q, vc_scale) = quantize_symmetric(&value_conv_out);
        let mut vfc1_i32 = vec![0i32; 128];
        matvec_int8_neon(&self.value_fc1_weight, &vc_q, &mut vfc1_i32, 128, hw);
        let value_hidden: Vec<f32> = (0..128)
            .map(|i| (vfc1_i32[i] as f32 * (vc_scale * self.value_fc1_scale) + self.value_fc1_bias[i]).max(0.0))
            .collect();

        // Value FC2
        let (vh_q, vh_scale) = quantize_symmetric(&value_hidden);
        let mut vfc2_i32 = vec![0i32; 1];
        matvec_int8_neon(&self.value_fc2_weight, &vh_q, &mut vfc2_i32, 1, 128);
        let value = (vfc2_i32[0] as f32 * (vh_scale * self.value_fc2_scale) + self.value_fc2_bias[0]).tanh();

        crate::mcts::search::NnOutput { policy, value }
    }
}

impl crate::nn::inference::NnEvaluator for Int8Evaluator {
    fn evaluate(&self, features: &[f32; crate::board::INPUT_SIZE]) -> crate::mcts::search::NnOutput {
        self.forward(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let data: Vec<f32> = (-10..10).map(|x| x as f32 * 0.1).collect();
        let (quantized, scale) = quantize_symmetric(&data);
        let recovered = dequantize(&quantized, scale);

        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.02, "orig={}, rec={}", orig, rec);
        }
    }

    #[test]
    fn test_matvec_int8() {
        let weight: Vec<i8> = vec![1, 2, 3, 4, 5, 6]; // 2x3
        let input: Vec<i8> = vec![1, 1, 1];
        let mut output = vec![0i32; 2];

        matvec_int8_neon(&weight, &input, &mut output, 2, 3);
        assert_eq!(output[0], 6);  // 1+2+3
        assert_eq!(output[1], 15); // 4+5+6
    }

    #[test]
    fn test_int8_evaluator_runs() {
        use crate::nn::inference::NnEvaluator;
        let weights = crate::nn::weights::NetworkWeights::random(2, 16);
        let evaluator = Int8Evaluator::from_weights(&weights);
        let input = [0.1f32; crate::board::INPUT_SIZE];
        let output = evaluator.evaluate(&input);
        assert_eq!(output.policy.len(), 1858);
        assert!(output.value >= -1.0 && output.value <= 1.0);
    }
}

/// f32 reference implementations of neural network layers.
/// These are correct but slow — the SIMD module provides optimized versions.

/// 2D convolution (NCHW format).
///
/// input: [in_c, h, w]
/// weight: [out_c, in_c, kh, kw]
/// bias: [out_c]
/// Returns: [out_c, h, w] (with padding to preserve spatial dims)
pub fn conv2d_f32(
    input: &[f32],
    in_channels: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel_size: usize,
    padding: usize,
) -> Vec<f32> {
    let out_h = height + 2 * padding - kernel_size + 1;
    let out_w = width + 2 * padding - kernel_size + 1;
    let mut output = vec![0.0f32; out_channels * out_h * out_w];

    for oc in 0..out_channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = bias[oc];

                for ic in 0..in_channels {
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh + kh;
                            let iw = ow + kw;

                            // Check padding bounds
                            let ih_orig = ih as isize - padding as isize;
                            let iw_orig = iw as isize - padding as isize;

                            if ih_orig >= 0
                                && ih_orig < height as isize
                                && iw_orig >= 0
                                && iw_orig < width as isize
                            {
                                let input_idx =
                                    ic * height * width + ih_orig as usize * width + iw_orig as usize;
                                let weight_idx = oc * in_channels * kernel_size * kernel_size
                                    + ic * kernel_size * kernel_size
                                    + kh * kernel_size
                                    + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }

    output
}

/// Batch normalization (inference mode).
///
/// x: [channels, h, w] — modified in place
/// mean, var: running statistics
/// weight (gamma), bias (beta): affine parameters
/// epsilon = 1e-5
pub fn batch_norm_f32(
    x: &mut [f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
) {
    let channels = mean.len();
    let spatial = x.len() / channels;
    let eps = 1e-5f32;

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = weight[c] * inv_std;
        let offset = bias[c] - mean[c] * scale;

        let start = c * spatial;
        let end = start + spatial;
        for i in start..end {
            x[i] = x[i] * scale + offset;
        }
    }
}

/// ReLU activation (in-place).
pub fn relu_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Fully connected layer.
///
/// input: [in_features]
/// weight: [out_features, in_features]
/// bias: [out_features]
/// Returns: [out_features]
pub fn linear_f32(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let in_features = input.len();
    let out_features = bias.len();

    let mut output = vec![0.0f32; out_features];

    for o in 0..out_features {
        let mut sum = bias[o];
        for i in 0..in_features {
            sum += weight[o * in_features + i] * input[i];
        }
        output[o] = sum;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity() {
        // 1x1 conv should act like a per-channel linear transform
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 channel, 2x2
        let weight = vec![2.0]; // 1 output, 1 input, 1x1
        let bias = vec![0.5];

        let output = conv2d_f32(&input, 1, 2, 2, &weight, &bias, 1, 1, 0);
        assert_eq!(output, vec![2.5, 4.5, 6.5, 8.5]);
    }

    #[test]
    fn test_relu() {
        let mut x = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        relu_f32(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_linear() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3 identity-ish
        let bias = vec![0.1, 0.2];

        let output = linear_f32(&input, &weight, &bias);
        assert!((output[0] - 1.1).abs() < 1e-6);
        assert!((output[1] - 2.2).abs() < 1e-6);
    }
}

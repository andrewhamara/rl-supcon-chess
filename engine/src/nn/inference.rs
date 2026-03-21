use crate::board::INPUT_SIZE;
use crate::mcts::search::NnOutput;
use crate::nn::weights::NetworkWeights;
use crate::nn::layers;
use crate::policy::POLICY_SIZE;

/// Trait for neural network evaluation.
/// Implementations can be f32 reference, int8 SIMD, or random (for testing).
pub trait NnEvaluator: Send + Sync {
    fn evaluate(&self, features: &[f32; INPUT_SIZE]) -> NnOutput;

    /// Batch evaluation for efficiency. Default falls back to sequential.
    fn evaluate_batch(&self, features: &[[f32; INPUT_SIZE]]) -> Vec<NnOutput> {
        features.iter().map(|f| self.evaluate(f)).collect()
    }
}

/// Random evaluator for testing and initial self-play.
pub struct RandomEvaluator;

impl NnEvaluator for RandomEvaluator {
    fn evaluate(&self, _features: &[f32; INPUT_SIZE]) -> NnOutput {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Uniform random policy
        let policy: Vec<f32> = (0..POLICY_SIZE).map(|_| rng.gen::<f32>()).collect();
        // Random value
        let value: f32 = rng.gen::<f32>() * 2.0 - 1.0;

        NnOutput { policy, value }
    }
}

/// f32 reference inference engine (no SIMD, for correctness testing).
pub struct F32Evaluator {
    weights: NetworkWeights,
}

impl F32Evaluator {
    pub fn new(weights: NetworkWeights) -> Self {
        F32Evaluator { weights }
    }

    /// Forward pass through the network.
    fn forward(&self, input: &[f32; INPUT_SIZE]) -> NnOutput {
        let w = &self.weights;

        // Input: [21, 8, 8] (NCHW)
        // Stem: Conv2d(21, 128, 3) -> BN -> ReLU
        let mut x = layers::conv2d_f32(
            input, 21, 8, 8,
            &w.stem_conv_weight, &w.stem_conv_bias,
            w.num_filters, 3, 1,
        );
        layers::batch_norm_f32(&mut x, &w.stem_bn_mean, &w.stem_bn_var, &w.stem_bn_weight, &w.stem_bn_bias);
        layers::relu_f32(&mut x);

        // Residual blocks
        for block in &w.residual_blocks {
            let residual = x.clone();

            // Conv1 -> BN -> ReLU
            x = layers::conv2d_f32(
                &x, w.num_filters, 8, 8,
                &block.conv1_weight, &block.conv1_bias,
                w.num_filters, 3, 1,
            );
            layers::batch_norm_f32(&mut x, &block.bn1_mean, &block.bn1_var, &block.bn1_weight, &block.bn1_bias);
            layers::relu_f32(&mut x);

            // Conv2 -> BN
            x = layers::conv2d_f32(
                &x, w.num_filters, 8, 8,
                &block.conv2_weight, &block.conv2_bias,
                w.num_filters, 3, 1,
            );
            layers::batch_norm_f32(&mut x, &block.bn2_mean, &block.bn2_var, &block.bn2_weight, &block.bn2_bias);

            // Skip connection + ReLU
            for (xi, ri) in x.iter_mut().zip(residual.iter()) {
                *xi += ri;
            }
            layers::relu_f32(&mut x);
        }

        // Policy head: Conv2d(128, 32, 1) -> BN -> ReLU -> Linear(2048, 1858)
        let mut policy_x = layers::conv2d_f32(
            &x, w.num_filters, 8, 8,
            &w.policy_conv_weight, &w.policy_conv_bias,
            32, 1, 0,
        );
        layers::batch_norm_f32(
            &mut policy_x,
            &w.policy_bn_mean, &w.policy_bn_var,
            &w.policy_bn_weight, &w.policy_bn_bias,
        );
        layers::relu_f32(&mut policy_x);
        let policy = layers::linear_f32(&policy_x, &w.policy_fc_weight, &w.policy_fc_bias);

        // Value head: Conv2d(128, 1, 1) -> BN -> ReLU -> Linear(64, 128) -> ReLU -> Linear(128, 1) -> Tanh
        let mut value_x = layers::conv2d_f32(
            &x, w.num_filters, 8, 8,
            &w.value_conv_weight, &w.value_conv_bias,
            1, 1, 0,
        );
        layers::batch_norm_f32(
            &mut value_x,
            &w.value_bn_mean, &w.value_bn_var,
            &w.value_bn_weight, &w.value_bn_bias,
        );
        layers::relu_f32(&mut value_x);
        let mut value_hidden = layers::linear_f32(&value_x, &w.value_fc1_weight, &w.value_fc1_bias);
        layers::relu_f32(&mut value_hidden);
        let value_out = layers::linear_f32(&value_hidden, &w.value_fc2_weight, &w.value_fc2_bias);
        let value = value_out[0].tanh();

        NnOutput { policy, value }
    }
}

impl NnEvaluator for F32Evaluator {
    fn evaluate(&self, features: &[f32; INPUT_SIZE]) -> NnOutput {
        self.forward(features)
    }
}

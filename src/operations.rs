//! Core neural network operations for the transformer.

use rayon::prelude::*;
use std::f32::consts::PI;

/// Apply RMS normalization to a vector.
///
/// RMSNorm normalizes the input by dividing by the root mean square,
/// then scales and shifts using learned parameters.
///
/// # Arguments
/// * `output` - Output buffer (will be modified)
/// * `input` - Input vector to normalize
/// * `weight` - Scale parameters (learned)
/// * `size` - Size of the vectors
pub fn rmsnorm(output: &mut [f32], input: &[f32], weight: &[f32], size: usize) {
    debug_assert_eq!(output.len(), size);
    debug_assert_eq!(input.len(), size);
    debug_assert_eq!(weight.len(), size);

    // Calculate sum of squares
    let ss: f32 = input.par_iter().map(|&x| x * x).sum();
    let ss = ss / size as f32 + 1e-5;
    let ss = 1.0 / ss.sqrt();

    // Normalize and scale
    output
        .par_iter_mut()
        .zip(input.par_iter().zip(weight.par_iter()))
        .for_each(|(out, (&x, &w))| {
            *out = w * (ss * x);
        });
}

/// Apply softmax to a vector in-place.
///
/// Converts logits to probabilities using the softmax function.
/// Uses numerical stability by subtracting the maximum value.
///
/// # Arguments
/// * `x` - Input/output vector (modified in-place)
/// * `size` - Size of the vector
pub fn softmax(x: &mut [f32], size: usize) {
    debug_assert_eq!(x.len(), size);

    // Find max value for numerical stability
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Exp and sum
    let mut sum = 0.0;
    for val in x.iter_mut().take(size) {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    let sum_recip = sum.recip();
    x.iter_mut().take(size).for_each(|val| *val *= sum_recip);
}

/// Matrix multiplication: output = weight @ input
///
/// Performs C = A @ B where A is (d,n), B is (n,), C is (d,)
/// This is the most compute-intensive operation in the transformer.
///
/// # Arguments
/// * `output` - Output vector of size d (will be modified)
/// * `input` - Input vector of size n
/// * `weight` - Weight matrix of size (d,n) in row-major order
/// * `n` - Input dimension
/// * `d` - Output dimension
pub fn matmul(output: &mut [f32], input: &[f32], weight: &[f32], n: usize, d: usize) {
    debug_assert_eq!(output.len(), d);
    debug_assert_eq!(input.len(), n);
    debug_assert_eq!(weight.len(), d * n);

    output.par_iter_mut().enumerate().for_each(|(i, out)| {
        let row_start = i * n;
        let row = &weight[row_start..row_start + n];
        *out = row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum();
    });
}

/// Generate rotary position embeddings (RoPE) frequencies.
///
/// RoPE (Rotary Position Embedding) rotates queries and keys based on position.
/// This function precomputes the frequencies used for rotation.
///
/// # Arguments
/// * `dim` - Model dimension
/// * `seq_len` - Maximum sequence length
/// * `theta` - Base for frequency calculation (usually 10000.0)
///
/// # Returns
/// Vector of complex frequencies for RoPE
pub fn precompute_freqs_cis(dim: usize, seq_len: usize, theta: f32) -> Vec<(f32, f32)> {
    let mut freqs_cis = Vec::with_capacity(seq_len * (dim / 2));

    for pos in 0..seq_len {
        for i in (0..dim).step_by(2) {
            let head_dim = i / 2;
            let freq = theta.powf(-(head_dim as f32) / (dim as f32));
            let val = (pos as f32) * freq;
            let (sin_val, cos_val) = val.sin_cos();
            freqs_cis.push((cos_val, sin_val));
        }
    }

    freqs_cis
}

/// Apply rotary position embedding to a vector.
///
/// Rotates pairs of elements in the vector using precomputed frequencies.
/// This implements the RoPE (Rotary Position Embedding) mechanism.
///
/// # Arguments
/// * `vec` - Vector to rotate (modified in-place)
/// * `freqs_cis` - Precomputed rotation frequencies
/// * `pos` - Position in sequence
/// * `dim` - Model dimension
pub fn apply_rotary_emb(vec: &mut [f32], freqs_cis: &[(f32, f32)], pos: usize, dim: usize) {
    debug_assert_eq!(vec.len(), dim);

    for i in (0..dim).step_by(2) {
        let head_dim = i / 2;
        let idx = pos * (dim / 2) + head_dim;
        let (fcr, fci) = freqs_cis[idx];

        let v0 = vec[i];
        let v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

/// Swish activation function (SiLU): x * sigmoid(x)
///
/// Used in the SwiGLU activation in feed-forward layers.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// Swish activation of x
pub fn swish(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

/// Efficient parallel sum of a slice.
///
/// # Arguments
/// * `slice` - Slice to sum
///
/// # Returns
/// Sum of all elements
pub fn parallel_sum(slice: &[f32]) -> f32 {
    slice.par_iter().sum()
}

/// Element-wise vector addition with scaling: a = a + b * scale
///
/// # Arguments
/// * `a` - First vector (modified in-place)
/// * `b` - Second vector
/// * `scale` - Scaling factor for b
pub fn add_scaled(a: &mut [f32], b: &[f32], scale: f32) {
    debug_assert_eq!(a.len(), b.len());

    a.par_iter_mut()
        .zip(b.par_iter())
        .for_each(|(a_val, &b_val)| {
            *a_val += b_val * scale;
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let mut output = vec![0.0; 4];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5];

        rmsnorm(&mut output, &input, &weight, 4);

        // Check that output has correct length and is normalized
        assert_eq!(output.len(), 4);
        // RMS should be approximately sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // Then each element is weight[i] * (input[i] / rms)
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / 4.0 + 1e-5).sqrt();
        for (i, &out) in output.iter().enumerate() {
            let expected = weight[i] * (input[i] / rms);
            assert!((out - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x, 3);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that values are positive and in descending order
        assert!(x[0] > 0.0 && x[1] > 0.0 && x[2] > 0.0);
        assert!(x[0] < x[1] && x[1] < x[2]); // Since input was ascending
    }

    #[test]
    fn test_matmul() {
        let mut output = vec![0.0; 2];
        let input = vec![1.0, 2.0, 3.0];
        // 2x3 weight matrix: [[1, 2, 3], [4, 5, 6]]
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        matmul(&mut output, &input, &weight, 3, 2);

        // output[0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        // output[1] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((output[0] - 14.0).abs() < 1e-6);
        assert!((output[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_swish() {
        let x = 1.0;
        let result = swish(x);
        let expected = x * (1.0 / (1.0 + (-x).exp()));
        assert!((result - expected).abs() < 1e-6);
    }
}

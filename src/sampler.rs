//! Token sampling strategies for text generation.

use rand::prelude::*;
use rand_pcg::Pcg64;

/// ProbIndex struct used for sorting probabilities during top-p sampling.
#[derive(Debug, Clone, Copy)]
pub struct ProbIndex {
    /// Probability value
    pub prob: f32,
    /// Index of the token
    pub index: usize,
}

/// Token sampler for controlling text generation.
///
/// Implements various sampling strategies: greedy, multinomial, and top-p (nucleus) sampling.
#[derive(Debug)]
pub struct Sampler {
    /// Vocabulary size
    vocab_size: usize,
    /// Temperature for controlling randomness (0.0 = greedy)
    temperature: f32,
    /// Top-p (nucleus) sampling threshold (0.0 = disabled)
    topp: f32,
    /// Random number generator
    rng: Pcg64,
    /// Buffer for top-p sampling (reused to avoid allocations)
    probindex: Vec<ProbIndex>,
}

impl Sampler {
    /// Create a new sampler with the specified parameters.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary
    /// * `temperature` - Sampling temperature (0.0 = greedy, higher = more random)
    /// * `topp` - Top-p threshold (0.0 = disabled, 0.9 = common value)
    /// * `rng_seed` - Random seed for reproducible results
    ///
    /// # Returns
    /// Configured sampler
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(rng_seed);
        let probindex = vec![ProbIndex { prob: 0.0, index: 0 }; vocab_size];

        Self {
            vocab_size,
            temperature,
            topp,
            rng,
            probindex,
        }
    }

    /// Sample a token from the given logits.
    ///
    /// Applies the configured sampling strategy (greedy, temperature, top-p)
    /// to select the next token from the logits distribution.
    ///
    /// # Arguments
    /// * `logits` - Raw logits from the model (will be modified in-place)
    ///
    /// # Returns
    /// Selected token ID
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        debug_assert_eq!(logits.len(), self.vocab_size);

        if self.temperature == 0.0 {
            // Greedy sampling: take the token with highest probability
            Self::argmax(logits)
        } else {
            // Apply temperature scaling
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }

            // Convert to probabilities with softmax
            self.softmax(logits);

            // Sample from the distribution
            let coin: f32 = self.rng.gen();
            if self.topp <= 0.0 || self.topp >= 1.0 {
                // Standard multinomial sampling
                self.multinomial_sample(logits, coin)
            } else {
                // Top-p (nucleus) sampling
                self.topp_sample(logits, coin)
            }
        }
    }

    /// Greedy argmax sampling (select highest probability token).
    fn argmax(probabilities: &[f32]) -> usize {
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Multinomial sampling from a probability distribution.
    fn multinomial_sample(&mut self, probabilities: &[f32], coin: f32) -> usize {
        let mut cdf = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cdf += prob;
            if coin < cdf {
                return i;
            }
        }
        // Fallback for numerical precision issues
        probabilities.len() - 1
    }

    /// Top-p (nucleus) sampling.
    ///
    /// Samples from the smallest set of tokens whose cumulative probability
    /// exceeds the top-p threshold.
    fn topp_sample(&mut self, probabilities: &[f32], coin: f32) -> usize {
        let cutoff = (1.0 - self.topp) / (self.vocab_size - 1) as f32;

        // Filter and sort probabilities above cutoff
        let mut n0 = 0;
        for i in 0..self.vocab_size {
            if probabilities[i] >= cutoff {
                self.probindex[n0].index = i;
                self.probindex[n0].prob = probabilities[i];
                n0 += 1;
            }
        }

        // Sort by probability (descending)
        self.probindex[..n0].sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

        // Find truncation point where cumulative probability exceeds topp
        let mut cumulative_prob = 0.0;
        let mut last_idx = n0 - 1;
        for i in 0..n0 {
            cumulative_prob += self.probindex[i].prob;
            if cumulative_prob > self.topp {
                last_idx = i;
                break;
            }
        }

        // Sample from the truncated distribution
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;
        for i in 0..=last_idx {
            cdf += self.probindex[i].prob;
            if r < cdf {
                return self.probindex[i].index;
            }
        }

        // Fallback
        self.probindex[last_idx].index
    }

    /// In-place softmax computation.
    fn softmax(&self, x: &mut [f32]) {
        // Find max value for numerical stability
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Exp and sum
        let mut sum = 0.0;
        for val in x.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        // Normalize
        let sum_recip = sum.recip();
        for val in x.iter_mut() {
            *val *= sum_recip;
        }
    }

    /// Update sampler parameters.
    ///
    /// # Arguments
    /// * `temperature` - New temperature value
    /// * `topp` - New top-p value
    pub fn update_params(&mut self, temperature: f32, topp: f32) {
        self.temperature = temperature;
        self.topp = topp;
    }

    /// Reset the random number generator with a new seed.
    ///
    /// # Arguments
    /// * `seed` - New random seed
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Pcg64::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(10, 0.0, 0.9, 42);
        let mut logits = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let token = sampler.sample(&mut logits);
        assert_eq!(token, 1); // Index of highest logit
    }

    #[test]
    fn test_softmax() {
        let mut sampler = Sampler::new(3, 1.0, 0.9, 42);
        let mut x = [1.0, 2.0, 3.0];

        sampler.softmax(&mut x);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that values are in ascending order (since inputs were)
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn test_multinomial_sampling() {
        let mut sampler = Sampler::new(3, 1.0, 0.0, 42); // topp = 0 disables top-p
        let mut logits = [0.0, 0.0, 0.0]; // Will become [1/3, 1/3, 1/3] after softmax

        // Sample multiple times to test distribution
        let mut counts = [0; 3];
        for _ in 0..1000 {
            let token = sampler.sample(&mut logits);
            counts[token] += 1;
        }

        // Each should be sampled roughly equally
        for &count in counts.iter() {
            assert!(count > 200 && count < 500); // Allow some variance
        }
    }

    #[test]
    fn test_topp_sampling() {
        let mut sampler = Sampler::new(4, 1.0, 0.5, 42); // topp = 0.5
        let mut logits = [1.0, 1.0, 1.0, 1.0]; // Equal probabilities

        // With topp=0.5, should only sample from top tokens that sum to > 0.5
        let mut counts = [0; 4];
        for _ in 0..1000 {
            let token = sampler.sample(&mut logits);
            counts[token] += 1;
        }

        // Should only sample from first 2-3 tokens due to topp
        assert_eq!(counts[3], 0); // Last token should never be sampled
    }
}

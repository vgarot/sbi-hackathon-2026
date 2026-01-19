"""Solution for generating synthetic data from NLE estimator"""

# NLE estimator learns p(x|theta), so we condition on theta and sample x
# This is the reverse of NPE, where we condition on x and sample theta

x_synthetic_nle = nle_net.sample((num_synthetic,), condition=theta_condition).squeeze(1)

print(f"Synthetic data shape: {x_synthetic_nle.shape}")  # Should be [100, 20]

"""Solution for MMD-based misspecification detection exercise"""

from sbi.diagnostics.misspecification import calc_misspecification_mmd

# Test well-specified observation
p_value_mmd_wellspec, (mmds_baseline_wellspec, mmd_wellspec) = calc_misspecification_mmd(
    inference=trainer,
    x_obs=x_o_wellspec,
    x=x_val,
    mode="embedding"  # Use learned embedding space
)
print(f"Well-specified observation (MMD):")
print(f"  p-value: {p_value_mmd_wellspec:.4f}")
print(f"  MMD: {mmd_wellspec:.6f}")

# Test misspecified observation
p_value_mmd_misspec, (mmds_baseline_misspec, mmd_misspec) = calc_misspecification_mmd(
    inference=trainer,
    x_obs=x_o_misspec,
    x=x_val,
    mode="embedding"
)
print(f"\nMisspecified observation (MMD):")
print(f"  p-value: {p_value_mmd_misspec:.4f}")
print(f"  MMD: {mmd_misspec:.6f}")

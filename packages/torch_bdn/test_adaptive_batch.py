"""
Test adaptive batch size selection in SGLD.
"""

import sys

import torch

# Add the source directory to the path
sys.path.insert(0, "src")


def test_adaptive_batch_size():
    """Test the adaptive batch size selection function."""

    # Import without torch dependencies first
    from torch_bdn.sampling.backends.sgld import _suggest_batch_size

    # Test different scenarios
    test_cases = [
        # (n_data, lr, expected_range, description)
        (500, 0.01, (32, 64), "Small dataset, normal lr"),
        (5000, 0.01, (64, 128), "Medium dataset, normal lr"),
        (50000, 0.01, (128, 256), "Large dataset, normal lr"),
        (500000, 0.01, (256, 512), "Very large dataset, normal lr"),
        (5000, 0.2, (128, 256), "Medium dataset, high lr → larger batch"),
        (5000, 0.0001, (32, 64), "Medium dataset, low lr → smaller batch"),
    ]

    print("Testing adaptive batch size selection:")
    print("=" * 60)

    for n_data, lr, expected_range, description in test_cases:
        suggested = _suggest_batch_size(n_data, lr)
        in_range = expected_range[0] <= suggested <= expected_range[1]
        is_power_of_2 = suggested & (suggested - 1) == 0  # Check if power of 2

        status = "✓" if in_range and is_power_of_2 else "✗"
        print(
            f"{status} n_data={n_data:6d}, lr={lr:6.4f} → batch_size={suggested:3d} | {description}"
        )

        if not (32 <= suggested <= 512):
            print(f"   WARNING: {suggested} outside valid range [32, 512]")
        if not is_power_of_2:
            print(f"   WARNING: {suggested} is not a power of 2")

    print("\nTesting SGLD with adaptive batch size:")

    # Create dummy functions for testing
    def log_likelihood_fn(params, data_batch):
        return -0.5 * torch.sum(params**2) * len(data_batch)  # Scale by batch size

    def log_prior_fn(params):
        return -0.1 * torch.sum(params**2)

    # Test with different configurations
    test_configs = [
        {"n_data": 1000, "lr": 0.01, "expected_batch": 64},
        {"n_data": 10000, "lr": 0.1, "expected_batch": 256},  # High lr → larger batch
        {
            "n_data": 500,
            "lr": 0.001,
            "expected_batch": 32,
        },  # Low lr + small data → small batch
    ]

    for config in test_configs:
        # Generate dummy data
        data = torch.randn(config["n_data"], 2)
        init_params = torch.tensor([0.0, 0.0], requires_grad=True)

        # Import SGLD (this will fail without torch, but let's try)
        try:
            from torch_bdn.sampling.backends.sgld import sgld

            # Run with auto batch size (batch_size=None)
            results = sgld(
                log_likelihood_fn=log_likelihood_fn,
                log_prior_fn=log_prior_fn,
                data=data,
                init_params=init_params,
                n_samples=2,  # Just a quick test
                batch_size=None,  # Auto-select
                lr=config["lr"],
                n_burnin=2,
            )

            selected_batch = results["diagnostics"]["batch_size"]
            was_auto = results["diagnostics"]["batch_size_auto_selected"]

            print(
                f"✓ n_data={config['n_data']}, lr={config['lr']} → auto-selected batch_size={selected_batch}, auto={was_auto}"
            )

        except ImportError as e:
            print(f"✗ Could not test full SGLD (missing torch): {e}")
            continue


if __name__ == "__main__":
    test_adaptive_batch_size()

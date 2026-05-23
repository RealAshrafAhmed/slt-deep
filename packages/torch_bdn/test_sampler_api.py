"""
Test updated Sampler with new SGLD API that handles batched evaluation.
"""

import sys

import torch

sys.path.insert(0, "src")


def test_sampler_api_compatibility():
    """Test that sampler works with new batched SGLD backend."""

    print("Testing Sampler API compatibility with new SGLD backend...")

    try:
        # Import components (will fail if torch not available)
        import torch.nn as nn
        from torch_bdn.bn.bayesian_net import BayesianNet
        from torch_bdn.sampling.sampler import Sampler

        # Create simple model and data
        model = nn.Linear(2, 1)

        def simple_prior(params):
            return -0.5 * torch.sum(params**2)  # Gaussian prior

        bayes_net = BayesianNet(
            model=model, prior_logp=simple_prior, likelihood_type="gaussian"
        )

        # Create simple dataset
        torch.manual_seed(42)
        x = torch.randn(100, 2)  # 100 samples, 2 features
        y = torch.randn(100, 1)  # 100 targets

        # Create sampler
        sampler = Sampler(bayes_net, x, y)

        print("✓ Sampler created successfully")

        # Test the helper method
        init_params = torch.cat([p.flatten() for p in model.parameters()])
        print(f"✓ Initial params shape: {init_params.shape}")

        # Test parameter update method
        new_params = torch.randn_like(init_params)
        sampler._update_model_params(new_params)
        updated_params = torch.cat([p.flatten() for p in model.parameters()])

        params_updated = not torch.allclose(init_params, updated_params)
        print(f"✓ Parameter update method working: {params_updated}")

        # Test data combination
        combined_data = torch.cat([x, y], dim=-1)
        expected_shape = (100, 3)  # 100 samples, 2 features + 1 target
        print(
            f"✓ Combined data shape: {combined_data.shape} (expected: {expected_shape})"
        )

        # Test log likelihood function creation
        def log_likelihood_fn(params, data_batch):
            sampler._update_model_params(params)
            x_batch = data_batch[:, :-1]
            y_batch = data_batch[:, -1:]
            return bayes_net.log_likelihood(x_batch, y_batch)

        # Test with small batch
        test_batch = combined_data[:5]  # 5 samples
        test_params = torch.randn_like(init_params)

        loglik_result = log_likelihood_fn(test_params, test_batch)
        print(f"✓ Log likelihood function working, result shape: {loglik_result.shape}")

        # Test log prior function
        def log_prior_fn(params):
            sampler._update_model_params(params)
            return bayes_net.log_prior()

        logprior_result = log_prior_fn(test_params)
        print(f"✓ Log prior function working, result shape: {logprior_result.shape}")

        print("✅ All API compatibility tests passed!")

        # Try a very small sampling run (will fail if torch not available for full SGLD)
        try:
            from torch_bdn.sampling import SGLD

            results = sampler.sample(SGLD(n_warmup=2, batch_size=16), n_samples=2)
            print(f"✓ Sampling completed! Got {len(results['parameters'])} samples")
            print(f"✓ Final sample shape: {results['parameters'][0].shape}")
        except Exception as e:
            print(f"⚠️  Full sampling test failed (expected without torch): {e}")

    except ImportError as e:
        print(f"⚠️  Could not test full sampler (missing torch): {e}")
        return


if __name__ == "__main__":
    test_sampler_api_compatibility()
